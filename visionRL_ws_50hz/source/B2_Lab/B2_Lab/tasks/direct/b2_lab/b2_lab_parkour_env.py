"""Extreme Parkour env for the B2 quadruped (IsaacLab direct workflow).

Subclass of B2LabEnv that adds, gated by cfg.parkour_mode:
  * a forward-facing depth TiledCamera (student / phase-2 perception),
  * a goal-heading + commanded-speed task (target_yaw / delta_yaw / goal_dir),
  * the Extreme-Parkour observation layout on the "policy"/"critic_obs" keys,
  * a "depth" entry in the observation dict for the vision phase.

The base B2LabEnv is untouched except for registering the parkour reward
container. Reference: Cheng et al., "Extreme Parkour with Legged Robots".
"""

from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, RayCaster, TiledCamera
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from .b2_lab_env import B2LabEnv, torch_rand_float
from .b2_lab_parkour_env_cfg import B2LabParkourEnvCfg


class B2LabParkourEnv(B2LabEnv):
    cfg: B2LabParkourEnvCfg

    # --------------------------------------------------------------- scene setup
    def _setup_scene(self):
        # NOTE: sensors must be added BEFORE scene.clone_environments(), so we
        # reproduce the base setup here and insert the depth camera in the middle
        # rather than calling super() (which clones before we could add it).
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner
        self._foot_scanner_fl = RayCaster(self.cfg.foot_scanner_FL)
        self.scene.sensors["foot_scanner_fl"] = self._foot_scanner_fl
        self._foot_scanner_fr = RayCaster(self.cfg.foot_scanner_FR)
        self.scene.sensors["foot_scanner_fr"] = self._foot_scanner_fr
        self._foot_scanner_rl = RayCaster(self.cfg.foot_scanner_RL)
        self.scene.sensors["foot_scanner_rl"] = self._foot_scanner_rl
        self._foot_scanner_rr = RayCaster(self.cfg.foot_scanner_RR)
        self.scene.sensors["foot_scanner_rr"] = self._foot_scanner_rr

        # depth camera (parkour student perception); skipped for phase-1 teacher.
        self._has_depth = (
            getattr(self.cfg, "parkour_mode", False)
            and getattr(self.cfg, "enable_depth", True)
            and self.cfg.depth_camera is not None
        )
        if self._has_depth:
            self._depth_camera = TiledCamera(self.cfg.depth_camera)
            self.scene.sensors["depth_camera"] = self._depth_camera

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # --------------------------------------------------------------- buffers
    def _init_buffers(self):
        super()._init_buffers()
        n = self.num_envs
        g = self.cfg.goals
        # parkour task buffers
        self.pk_cmd_vel = torch.zeros(n, device=self.device)          # commanded forward speed
        self.cmd_vel = torch.zeros(n, device=self.device)             # exposed to reward container
        self.delta_yaw = torch.zeros(n, device=self.device)
        self.delta_next_yaw = torch.zeros(n, device=self.device)
        self.goal_dir_b = torch.zeros(n, 2, device=self.device)       # unit goal dir in base xy
        self.last_torques = torch.zeros(n, self.num_actions, device=self.device)

        # world-space forward goal waypoints
        self.num_goals = g.num_goals
        self.goal_positions = torch.zeros(n, g.num_goals, 3, device=self.device)
        self.cur_goal_idx = torch.zeros(n, dtype=torch.long, device=self.device)
        # forward offsets along +x (world), applied from each env's spawn origin
        self._goal_offsets = torch.zeros(g.num_goals, 3, device=self.device)
        self._goal_offsets[:, 0] = torch.arange(1, g.num_goals + 1, device=self.device) * g.spacing
        # per-foot edge flags (for feet_edge reward)
        self.feet_at_edge = torch.zeros(n, 4, dtype=torch.bool, device=self.device)

        # separate proprio-history buffer for the parkour layout (yaw commands
        # instead of velocity commands); kept apart from the base obs history.
        self.pk_obs_history_buf = torch.zeros(
            n, self.cfg.num_hist_pk, self.cfg.num_prop_pk, device=self.device
        )

        # undesired-contact bodies for the collision penalty: base + thighs + calves.
        thigh_ids, _ = self._contact_sensor.find_bodies([".*thigh.*"])
        self._penalised_contact_ids = list(self._base_id) + list(thigh_ids) + list(self._undesired_contact_body_ids)

    # --------------------------------------------------------------- commands
    def _resample_commands(self, env_ids):
        # Goals are position-based (set on reset); here we only resample the
        # commanded forward speed and keep the base velocity command consistent
        # so the gait clock / zero-command logic behaves.
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        rng = self.cfg.parkour_commands
        self.pk_cmd_vel[env_ids] = torch_rand_float(
            rng.speed_range[0], rng.speed_range[1], (n, 1), device=self.device
        ).squeeze(1)
        self._commands[env_ids, 0] = self.pk_cmd_vel[env_ids]
        self._commands[env_ids, 1] = 0.0
        self._commands[env_ids, 2] = 0.0

    # --------------------------------------------------------------- reset / goals
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None or (hasattr(env_ids, "numel") and env_ids.numel() == 0):
            return
        # Lay out forward goal waypoints along +x from each env's spawn origin.
        origins = self._terrain.env_origins[env_ids]                       # [k, 3]
        self.goal_positions[env_ids] = origins.unsqueeze(1) + self._goal_offsets.unsqueeze(0)
        self.cur_goal_idx[env_ids] = 0

    # --------------------------------------------------------------- goal update
    def _base_yaw(self):
        _, _, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        return wrap_to_pi(yaw)

    def _cur_goal(self, idx):
        return self.goal_positions[torch.arange(self.num_envs, device=self.device), idx]

    def _update_parkour_goals(self):
        base_pos = self._robot.data.root_pos_w[:, :2]
        base_yaw = self._base_yaw()

        cur_goal = self._cur_goal(self.cur_goal_idx)[:, :2]
        to_goal = cur_goal - base_pos                                       # [N,2] world xy
        dist = torch.norm(to_goal, dim=-1)

        # advance to the next waypoint when close enough
        reached = dist < self.cfg.goals.reach_threshold
        self.cur_goal_idx = torch.clamp(self.cur_goal_idx + reached.long(), max=self.num_goals - 1)

        # recompute after possible advance
        cur_goal = self._cur_goal(self.cur_goal_idx)[:, :2]
        to_goal = cur_goal - base_pos
        target_yaw = torch.atan2(to_goal[:, 1], to_goal[:, 0])
        self.delta_yaw = wrap_to_pi(target_yaw - base_yaw)

        next_idx = torch.clamp(self.cur_goal_idx + 1, max=self.num_goals - 1)
        next_goal = self._cur_goal(next_idx)[:, :2]
        to_next = next_goal - base_pos
        next_yaw = torch.atan2(to_next[:, 1], to_next[:, 0])
        self.delta_next_yaw = wrap_to_pi(next_yaw - base_yaw)

        # goal direction in the base frame = heading offset (cos, sin)
        self.goal_dir_b = torch.stack([torch.cos(self.delta_yaw), torch.sin(self.delta_yaw)], dim=-1)
        self.cmd_vel = self.pk_cmd_vel

        self._update_feet_at_edge()

    def _update_feet_at_edge(self):
        # Detect terrain edges under each foot from the per-ray height spread of
        # the foot scanner (foot_height_vec: [N, 4*25]). A large spread while the
        # foot is in contact indicates the foot is on/near an edge.
        fh = self.foot_height_vec.view(self.num_envs, 4, -1)                # [N,4,25]
        spread = fh.max(dim=-1).values - fh.min(dim=-1).values             # [N,4]
        self.feet_at_edge = (spread > self.cfg.goals.edge_threshold) & self.foot_contact_state

    def _get_dones(self):
        # Runs right after physics and before compute_reward, so parkour goal
        # buffers are fresh when the reward container reads them.
        died, time_out = super()._get_dones()
        self._update_parkour_goals()
        return died, time_out

    # --------------------------------------------------------------- depth
    def _get_depth_image(self):
        near, far = self.cfg.depth_near, self.cfg.depth_far
        out = self._depth_camera.data.output
        depth = out.get("distance_to_image_plane", None) if hasattr(out, "get") else out["distance_to_image_plane"]
        if depth is None:
            # camera not rendered yet (e.g. very first observation) -> return far plane
            return torch.ones(self.num_envs, self.cfg.depth_height, self.cfg.depth_width, device=self.device)
        if depth.dim() == 4:
            depth = depth.squeeze(-1)
        depth = torch.nan_to_num(depth, nan=far, posinf=far, neginf=far)
        depth = depth.clamp(near, far)
        depth = (depth - near) / (far - near)   # normalize to [0, 1]
        return depth

    # --------------------------------------------------------------- observations
    def _get_observations(self) -> dict:
        # Reuse the base pipeline for all sensor/privileged buffers.
        obs_dict = super()._get_observations()

        scandots = obs_dict["height_obs"]                                   # [N, 100]
        priv_latent = self.privileged_obs_buf                              # [N, 33]
        priv_explicit = self._robot.data.root_lin_vel_b * 2.0             # [N, 3]

        # Parkour proprioception (49): identical to base except the 3 command
        # slots hold [cmd_vel, delta_yaw, delta_next_yaw]. The depth backbone
        # predicts (delta_yaw, delta_next_yaw) and overwrites indices [6:8].
        pk_prop = torch.cat(
            [
                self._robot.data.root_ang_vel_b * 0.25,                    # 3
                self._robot.data.projected_gravity_b,                      # 3
                torch.stack([self.cmd_vel, self.delta_yaw, self.delta_next_yaw], dim=-1),  # 3
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,           # 12
                self._robot.data.joint_vel * 0.05,                         # 12
                self._actions,                                             # 12
                self.clock_inputs,                                         # 4
            ],
            dim=-1,
        )  # [N, 49]

        # Roll the parkour proprio history.
        self.pk_obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([pk_prop] * self.cfg.num_hist_pk, dim=1),
            torch.cat([self.pk_obs_history_buf[:, 1:], pk_prop.unsqueeze(1)], dim=1),
        )
        pk_hist_flat = self.pk_obs_history_buf.view(self.num_envs, -1)     # [N, 490]

        # Extreme-Parkour flat policy observation (675).
        policy_obs = torch.cat([pk_prop, scandots, priv_explicit, priv_latent, pk_hist_flat], dim=-1)

        obs_dict["policy"] = policy_obs
        obs_dict["critic_obs"] = policy_obs   # symmetric privileged critic
        obs_dict["prop_obs_parkour"] = pk_prop
        if self._has_depth:
            obs_dict["depth"] = self._get_depth_image()
        return obs_dict

    # --------------------------------------------------------------- bookkeeping
    def _post_physics_step(self):
        super()._post_physics_step()
        # Store torques for the delta_torques penalty (read on the next step,
        # which is when compute_reward runs relative to this update).
        self.last_torques = self._robot.data.applied_torque.clone()
