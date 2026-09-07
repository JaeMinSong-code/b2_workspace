from __future__ import annotations
import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.sensors import ContactSensor, RayCaster
from .b2_lab_env_cfg import B2LabFlatEnvCfg
import numpy as np
from isaaclab.utils.math import quat_apply_inverse, quat_apply, wrap_to_pi, quat_apply_yaw
import omni.usd
from pxr import UsdGeom, Gf, Sdf
from .obstacle_ladder import spawn_ladder_boxes_per_env
import time


def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(shape, device=device) + lower


class B2LabEnv(DirectRLEnv):
    cfg: B2LabFlatEnvCfg

    def __init__(self, cfg: B2LabFlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._parse_cfg(self.cfg)
        self._init_buffers()
        self._prepare_reward_function()
        self._prepare_cost_function()
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self.scene.write_data_to_sim()
        self.sim.forward()
        # if self.cfg.debug_viz:
        #     self._init_goal_viz()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner
        # 발 순서 [FL, FR, RL, RR] : find_bodies(".*foot.*") 및 gait/obs 인덱싱과 일치
        self._foot_scanner_fl = RayCaster(self.cfg.foot_scanner_FL)
        self.scene.sensors["foot_scanner_fl"] = self._foot_scanner_fl
        self._foot_scanner_fr = RayCaster(self.cfg.foot_scanner_FR)
        self.scene.sensors["foot_scanner_fr"] = self._foot_scanner_fr
        self._foot_scanner_rl = RayCaster(self.cfg.foot_scanner_RL)
        self.scene.sensors["foot_scanner_rl"] = self._foot_scanner_rl
        self._foot_scanner_rr = RayCaster(self.cfg.foot_scanner_RR)
        self.scene.sensors["foot_scanner_rr"] = self._foot_scanner_rr
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # spawn_ladder_boxes_per_env(
        #     num_envs=self.num_envs,
        #     base_x=2.0, base_y=0.0, base_z=0.0,
        #     ladder_width=1.45,
        #     ladder_height=2.2,
        #     rail_thickness=0.08,
        #     rung_thickness=0.07,
        #     rung_count=12,
        #     root_name="Ladder",
        #     pitch_deg=15.0,
        # )

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
        self.joint_pos_target = self._processed_actions.clone()

        # World frame에서 발 위치 가져오기 (이름 기반 foot body id, 순서 [FL,FR,RL,RR])
        self.foot_positions = self._robot.data.body_pos_w[:, self._feet_ids, :]
        # Body frame으로 변환
        base_quat = self._robot.data.root_quat_w  # [num_envs, 4]
        base_pos = self._robot.data.root_pos_w    # [num_envs, 3]
        # 발 위치를 body frame으로 변환 (상대 위치)
        foot_pos_relative = self.foot_positions - base_pos.unsqueeze(1)  # [num_envs, 4, 3]
        # 쿼터니언을 발 개수(4) 축으로 확장해 배치 크기를 맞춤
        base_quat_expanded = base_quat.unsqueeze(1).expand(-1, foot_pos_relative.shape[1], -1).contiguous()
        self.foot_pos_b = quat_apply_inverse(base_quat_expanded, foot_pos_relative)  # [num_envs, 4, 3]
        # 평균 발 위치 및 속도 계산
        self.prev_foot_positions_mean = self.foot_positions_mean.clone()
        self.foot_positions_mean = torch.mean(self.foot_positions, dim=1)  # [num_envs, 3]
        self.foot_velocity_mean = (self.foot_positions_mean - self.prev_foot_positions_mean) / self.dt
        # 각 발의 개별 속도 계산 (optional)
        foot_velocities_w = (self.foot_positions - self.prev_foot_positions) / self.dt  # [num_envs, 4, 3]
        self.foot_velocities_b = quat_apply_inverse(base_quat_expanded, foot_velocities_w)  # [num_envs, 4, 3]

        # Contact force 처리
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        foot_forces_norm = torch.norm(net_contact_forces[:, :, self._feet_ids_contact], dim=-1)  # [num_envs, history_len, 4]
        max_foot_forces, _ = torch.max(foot_forces_norm, dim=1)  # [num_envs, 4] - history에서 최대값
        self.foot_force = net_contact_forces[:, -1, self._feet_ids_contact, :]
        self.foot_contact_state = max_foot_forces > 1.0

        # Contact time 업데이트
        contact_mask = self.foot_contact_state
        self.foot_contact_time = torch.where(
            contact_mask,
            self.foot_contact_time + self.dt,  # 접촉 중이면 시간 증가
            torch.zeros_like(self.foot_contact_time),  # 떨어지면 리셋
        )
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            base_quat = self._robot.data.root_quat_w
            forward = quat_apply(base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            heading_err = wrap_to_pi(self._heading_cmd - heading)  # [-pi, pi]
            yaw_cmd = 0.5 * heading_err
            yaw_cmd = torch.clip(yaw_cmd, -0.5, 0.5)
            deadband = 5.0 * np.pi / 180.0  # 5 deg in rad
            yaw_cmd = torch.where(torch.abs(heading_err) < deadband, torch.zeros_like(yaw_cmd), yaw_cmd)
            yaw_cmd = torch.where(self._yaw_stop_mask, torch.zeros_like(yaw_cmd), yaw_cmd)
            yaw_cmd = torch.where(torch.abs(yaw_cmd) < 0.05, torch.zeros_like(yaw_cmd), yaw_cmd)
            self._commands[:, 2] = yaw_cmd
        self._update_lin_vel_filter()

    def _post_physics_step(self):
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self._actions[:]
        self.prev_foot_positions = self.foot_positions.clone()

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions, env_ids=self._robot._ALL_INDICES)

    def _get_observations(self) -> dict:
        self._step_contact_targets()
        height_data = (
            self._height_scanner.data.pos_w[:, 2].unsqueeze(1)
            - self._height_scanner.data.ray_hits_w[..., 2]
            - 0.5
        ).clip(-1.0, 1.0)
        self.com_height = torch.mean(height_data, dim=-1, keepdim=True) + 0.5
        foot_scanners = {
            "fl": self._foot_scanner_fl,
            "fr": self._foot_scanner_fr,
            "rl": self._foot_scanner_rl,
            "rr": self._foot_scanner_rr,
        }

        self.footscanner_height_data, self.foot_height_vec = self.compute_foot_height_data(
            foot_scanners=foot_scanners,
            max_clearance=0.55,
            clip_range=(-1.0, 1.0),
        )
        obs = torch.cat(
            [
                self._robot.data.root_ang_vel_b * 0.25,                       # 3
                self._robot.data.projected_gravity_b,                  # 3
                self._commands * self.commands_scale,                                        # 3
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,  # 12
                self._robot.data.joint_vel * 0.05,                             # 12
                self._actions,                                         # 12
                self.clock_inputs,                                     # 4
            ],
            dim=-1,
        )  # total = 61

        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([obs] * self.cfg.num_history_len, dim=1),
            torch.cat(
                [self.obs_history_buf[:, 1:], obs.unsqueeze(1)],
                dim=1,
            ),
        )
        self.flattened_history_obs = self.obs_history_buf.view(self.num_envs, -1)

        self.privileged_obs_buf = torch.cat(
            (
                self.foot_contact_state,           # 4
                self.dynamic_fric_coeffs,           # 4
                self.actuator_stiffness_gains,      # 12
                self.actuator_damping_gains,        # 12
                self.com_height * 5.0,                    # 1
            ),
            dim=-1,
        )

        self.obs_buf = obs
        self.critic_obs = torch.cat(
            (
                self.obs_buf,
                self._robot.data.root_lin_vel_b * 2.0,
                self.privileged_obs_buf,
                self.foot_height_vec * 5.0,
            ),
            dim=-1,
        )
        # height_data = torch.zeros(self.num_envs, 289, device=self.device)
        # height_data = torch.zeros(self.num_envs, 441, device=self.device)
        observations = {
            "policy": self.actor_obs,
            "critic_obs": self.critic_obs,
            "prop_obs": self.obs_buf,
            "prop_obs_history": self.flattened_history_obs,
            "height_obs": height_data,
            "velocity_estimator_obs": self.flattened_history_obs,
            "velocity_estimator_target": self._robot.data.root_lin_vel_b,
            "priv_obs": self.flattened_history_obs,
            "priv_obs_target": self.privileged_obs_buf,
        }

        return observations

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self._max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        base_contact = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        # body_contact = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._body_contact_ids], dim=-1), dim=1)[0] > 1.0, dim=1)
        died = base_contact
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            m = int(self._max_episode_length)
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, low=max(1, int(0.5 * m)), high=m)
        self._actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.gait_indices[env_ids] = 0
        self.foot_contact_state[env_ids] = False
        self.prev_foot_positions_mean[env_ids] = 0.0
        self.foot_velocity_mean[env_ids] = 0.0
        self.gait_types[env_ids] = torch.randint(0, 1, (len(env_ids),), device=self.device)
        self.zero_hold_state[env_ids] = False
        self._resample_commands(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._lin_vel_xy_filt[env_ids] = self._robot.data.root_lin_vel_b[env_ids, :2]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        #-------------if zmp preveiw 관련 변수들 초기화-------------#
        root_xy = default_root_state[:, :2]
        self.zmp_wp_xy[env_ids]      = root_xy
        self.zmp_wp_xy_seq[env_ids]  = root_xy[:, None, :].repeat(1, self._zmp_wp_H, 1)
        self.zmp_ref_xy_seq[env_ids] = root_xy[:, None, :].repeat(1, self._zmp_ref_N, 1)
        self.prev_zero_command[env_ids] = False
        self.com_ref_xy_seq[env_ids]  = root_xy[:, None, :].repeat(1, self._zmp_ref_N, 1)
        self.com_ref_dxy_seq[env_ids] = 0.0
        
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # 보상 로깅
        reward_extras = dict()
        reward_extras["Episode_Reward"] = {}
        for key in self.episode_sums.keys():
            reward_extras["Episode_Reward"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.

        # 종료 로깅
        termination_extras = dict()
        termination_extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        termination_extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()

        # self.extras에 통합
        self.extras["log"] = dict()
        self.extras["log"].update(reward_extras)
        self.extras["log"].update(termination_extras)

        self._ep_track_sum[env_ids] = 0.0
        self._ep_track_count[env_ids] = 0.0

    def _step_contact_targets(self):
        gait_prev = self.gait_indices.clone()
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * 1.0, 1.0)
        durations = torch.full((self.num_envs, 1), 0.7, device=self.device)
        # trott = torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device).unsqueeze(0)
        walk = torch.tensor([0.0, 0.25, 0.5, 0.75], device=self.device).unsqueeze(0)
        # gait_offsets = torch.where(self.gait_types[:, None] == 0, trott, walk)
        gait_offsets = walk
        foot_indices = torch.remainder(self.gait_indices.unsqueeze(1) + gait_offsets, 1.0)
        self.foot_indices = foot_indices.clone()
        self.zero_command = (self._commands[:, :3].abs() < 0.05).all(dim=1)
        # #----------------- ZMP preview 업데이트 -----------------#
        # enter_zero = self.zero_command & (~self.prev_zero_command)
        # self._update_zmp_preview_from_cmd(gait_prev=gait_prev, enter_zero=enter_zero)
        # self._update_com_preview_from_zmp_ref(enter_zero=enter_zero)
        # self.prev_zero_command = self.zero_command.clone()
        # #----------------- ZMP preview 업데이트 -----------------#

        mask_zero = self.zero_command.unsqueeze(1)
        foot_at_zero = torch.abs(foot_indices) < 0.05

        self.zero_hold_state = torch.where(
            mask_zero,
            self.zero_hold_state | (mask_zero & foot_at_zero),  # 기존 래치 유지 + 새로운 래치
            torch.zeros_like(self.zero_hold_state)  # zero_command 해제시 모든 래치 초기화
        )

        self.foot_indices = torch.where(
            self.zero_hold_state,
            torch.zeros_like(self.foot_indices),  # foot_indices를 0으로 고정
            self.foot_indices
        )

        # clock inputs (sawtooth normalization)
        remainder = torch.remainder(self.foot_indices, 1.0)
        stance_mask = remainder < durations
        swing_mask = remainder > durations

        clock_inputs = torch.zeros_like(self.foot_indices)
        durations_exp = durations.expand_as(self.foot_indices)
        # stance phase
        clock_inputs[stance_mask] = remainder[stance_mask] * (0.5 / durations_exp[stance_mask])
        # swing phase
        clock_inputs[swing_mask] = 0.5 + (remainder[swing_mask] - durations_exp[swing_mask]) * (
            0.5 / (1.0 - durations_exp[swing_mask])
        )
        self.last_clock_inputs = self.clock_inputs.clone()
        self.clock_inputs = torch.sin(2 * np.pi * clock_inputs)

        # smoothing contact pattern
        kappa = self.cfg.rewards.kappa_gait_probs
        normal_dist = torch.distributions.normal.Normal(0, kappa)
        r = torch.remainder(self.foot_indices, 1.0)
        smoothing = (normal_dist.cdf(r) * (1.0 - normal_dist.cdf(r - 0.5))
                     + normal_dist.cdf(r - 1.0) * (1.0 - normal_dist.cdf(r - 1.5)))
        self.desired_contact_states = smoothing

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        action = action.to(self.device)
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        self._pre_physics_step(action)
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)

            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()

            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs

        self.compute_reward()
        self.compute_cost()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            # IsaacLab 2.3.2: rerender_on_reset 은 deprecated → num_rerenders_on_reset 사용
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

        # apply interval events
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self._get_observations()

        # observation noise (앞부분: ang_vel3+grav3+cmd3+joint_pos+joint_vel = 9+2*num_actions 에만 적용)
        n_noisy = 9 + 2 * self.num_actions  # B2 = 9 + 24 = 33
        prop_obs_noisy = self.obs_buf["prop_obs"][:, :n_noisy]
        prop_obs_clean = self.obs_buf["prop_obs"][:, n_noisy:]
        if self.cfg.observation_noise_model:
            prop_obs_noisy = self._observation_noise_model(prop_obs_noisy)
        self.obs_buf["prop_obs"] = torch.cat([prop_obs_noisy, prop_obs_clean], dim=-1)

        # RSL-RL extras
        if "observations" not in self.extras:
            self.extras.update(self.obs_buf)

        self._post_physics_step()
        return self.obs_buf, self.rew_buf, self.cost_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _parse_cfg(self, cfg):
        self.dt = 1 / 50
        self.reward_scales = self.class_to_dict(self.cfg.rewards.scales)
        self.cost_scales = self.class_to_dict(self.cfg.costs.scales)
        self.command_ranges = self.class_to_dict(self.cfg.commands.ranges)
        self._max_episode_length_s = self.cfg.episode_length_s
        self._max_episode_length = np.ceil(self._max_episode_length_s / self.dt)

    def _compute_foot_shape_ids(self):
        """각 발(body)의 collision shape 인덱스 목록. USD별 shape 개수 차이에 robust
        (IsaacLab randomize_rigid_body_material 과 동일한 body→shape 매핑 방식)."""
        num_shapes_per_body = []
        for link_path in self._robot.root_physx_view.link_paths[0]:
            link_view = self._robot._physics_sim_view.create_rigid_body_view(link_path)
            num_shapes_per_body.append(link_view.max_shapes)
        shape_ids = []
        for body_id in self._feet_ids:
            start = sum(num_shapes_per_body[:body_id])
            end = start + num_shapes_per_body[body_id]
            shape_ids.extend(range(start, end))
        return shape_ids

    def _init_buffers(self):
        self.num_actions = gym.spaces.flatdim(self.single_action_space)  # B2 : 12
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._heading_cmd = torch.zeros(self.num_envs, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.frequencies = torch.zeros(self.num_envs, device=self.device)
        self.gait_indices = torch.zeros(self.num_envs, device=self.device)
        self.gait_types = torch.randint(0, 1, (self.num_envs,), device=self.device)
        self.clock_inputs = torch.ones(self.num_envs, 4, device=self.device)
        self.desired_contact_states = torch.ones(self.num_envs, 4, device=self.device)
        self.foot_contact_state = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.bool)  # 발 접촉 상태
        # articulation 과 contact sensor 의 body 순서가 다르므로 인덱스를 분리하고
        # 발 순서를 [FL,FR,RL,RR] 로 명시 고정한다(preserve_order).
        _FOOT_ORDER = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self._feet_ids, _ = self._robot.find_bodies(_FOOT_ORDER, preserve_order=True)                 # articulation 데이터용(body_pos/vel)
        self._feet_ids_contact, _ = self._contact_sensor.find_bodies(_FOOT_ORDER, preserve_order=True)  # contact 데이터용(net_forces)
        self._base_id, _ = self._contact_sensor.find_bodies("base_link")

        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies([".*thigh.*", ".*calf.*"])
        self._body_contact_ids, _ = self._contact_sensor.find_bodies("base_link")
        self.masses_tensor = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        # 발 4개의 collision shape 인덱스를 body→shape 매핑으로 정확히 계산 (USD 마다 shape 개수 상이)
        self._foot_shape_ids = self._compute_foot_shape_ids()
        self.static_fric_coeffs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.dynamic_fric_coeffs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.actuator_stiffness_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actuator_damping_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        mat = self._robot.root_physx_view.get_material_properties().to(self.device)  # [N, num_shapes, 3]
        self.static_fric_coeffs = mat[:, self._foot_shape_ids, 0]
        self.dynamic_fric_coeffs = mat[:, self._foot_shape_ids, 1]
        self.restitution_coeffs = mat[:, self._foot_shape_ids, 2]
        for _, actuator in self._robot.actuators.items():
            self.actuator_stiffness_gains = (self._robot.data.default_joint_stiffness - actuator.stiffness)
            self.actuator_damping_gains = (self._robot.data.default_joint_damping - actuator.damping)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.rew_buf_pos = torch.zeros(self.num_envs, device=self.device)
        self.rew_buf_neg = torch.zeros(self.num_envs, device=self.device)
        self.cost_buf = torch.zeros(self.num_envs, device=self.device)
        self.obs_buf = {}  # observation 딕셔너리 초기화
        self.com_height = torch.zeros(self.num_envs, 1, device=self.device)

        self.actor_obs = torch.zeros(self.num_envs, self.cfg.num_actor_obs, dtype=torch.float, device=self.device, requires_grad=False)
        self.heights_buf = torch.zeros(self.num_envs, self.cfg.num_scandots, dtype=torch.float, device=self.device, requires_grad=False)
        self.critic_obs = torch.zeros(self.num_envs, self.cfg.num_critic_obs, dtype=torch.float, device=self.device, requires_grad=False)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.cfg.num_privileged_obs, device=self.device, dtype=torch.float)
        self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.num_history_len, self.cfg.num_proprio, device=self.device, dtype=torch.float)
        self.flattened_history_obs = torch.zeros(self.num_envs, self.cfg.num_history_len * self.cfg.num_proprio, device=self.device, dtype=torch.float)

        self.foot_positions = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.prev_foot_positions = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.foot_positions_mean = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_foot_positions_mean = torch.zeros(self.num_envs, 3, device=self.device)
        self.foot_velocity_mean = torch.zeros(self.num_envs, 3, device=self.device)
        self.foot_pos_b = torch.zeros(self.num_envs, 4, 3, device=self.device) 
        self.foot_velocities_b = torch.zeros(self.num_envs, 4, 3, device=self.device)
        # B2 기립자세 nominal foot position (base frame), 순서 [FL,FR,RL,RR].
        # 기립자세(zero-action)로 세틀시킨 실측값 (raibert_foot_placement 보상 기준점).
        self.default_foot_pos_b = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.default_foot_pos_b[:] = torch.tensor([
            [ 0.427,  0.191, -0.452],   # FL
            [ 0.423, -0.192, -0.458],   # FR
            [-0.232,  0.193, -0.404],   # RL
            [-0.234, -0.192, -0.409],   # RR
        ], device=self.device)

        self.joint_pos_target = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_joint_pos_target = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_joint_pos_target = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.undesired_contacts = torch.zeros(self.num_envs, device=self.device)
        self.foot_force = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.foot_contact_time = torch.zeros((self.num_envs, 4), device=self.device)
        self.zero_command = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 제로 커맨드 래치: 발별로 스윙 종료(stance 진입) 이후 1로 고정
        self.zero_hold_state = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self._ep_start_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._ep_progress = torch.zeros(self.num_envs, device=self.device)
        # episode tracking accumulators
        self._ep_track_sum = torch.zeros(self.num_envs, device=self.device)      # 누적 점수
        self._ep_track_count = torch.zeros(self.num_envs, device=self.device)    # 누적 스텝 수
        self._ep_track_mean = torch.zeros(self.num_envs, device=self.device)     # reset 시 log용(선택)
        self.footscanner_height_data = torch.zeros(self.num_envs, 4, device=self.device)
        self.foot_height_vec = torch.zeros(self.num_envs, 4 * 25, device=self.device)
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)

        self._lin_vel_xy_filt = torch.zeros(self.num_envs, 2, device=self.device)
        self._yaw_stop_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.commands_scale = torch.tensor([2.0, 2.0, 0.25], device=self.device, requires_grad=False,)
        #zmp reference 관련
        self._init_zmp_preview_buffers(gait_freq=1.0, n_steps=4)

    def class_to_dict(self, obj) -> dict:
        if not hasattr(obj, "__dict__"):
            return obj
        result = {}
        for key in dir(obj):
            if key.startswith("_"):
                continue
            try:
                val = getattr(obj, key)
                if callable(val):
                    continue
                if isinstance(val, list):
                    element = []
                    for item in val:
                        element.append(self.class_to_dict(item))
                elif hasattr(val, "__dict__"):
                    element = self.class_to_dict(val)
                else:
                    element = val
                result[key] = element
            except (AttributeError, TypeError):
                continue
        return result

    def _prepare_reward_function(self):
        from .b2quad_reward import B2quadReward
        reward_containers = {"B2quadReward": B2quadReward}
        container_name = getattr(self.cfg, 'reward_container_name', 'B2quadReward')
        if hasattr(self.cfg, 'rewards') and hasattr(self.cfg.rewards, 'reward_container_name'):
            container_name = self.cfg.rewards.reward_container_name
        self.reward_container = reward_containers[container_name](self)

        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            if not hasattr(self.reward_container, '_reward_' + name):
                print(f"Warning: reward {'_reward_' + name} has nonzero coefficient but was not found!")
            else:
                self.reward_names.append(name)
                self.reward_functions.append(getattr(self.reward_container, '_reward_' + name))
        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.episode_sums_eval = {
            name: -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums_eval["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                      requires_grad=False)

    def _prepare_cost_function(self):
        from .b2quad_cost import B2quadCost
        cost_containers = {"B2quadCost": B2quadCost}
        self.cost_container = cost_containers[self.cfg.costs.cost_container_name](self)

        for key in list(self.cost_scales.keys()):
            scale = self.cost_scales[key]
            if scale == 0:
                self.cost_scales.pop(key)
        # prepare list of functions
        self.cost_functions = []
        self.cost_names = []
        for name, scale in self.cost_scales.items():
            if not hasattr(self.cost_container, '_cost_' + name):
                print(f"Warning: cost {'_cost_' + name} has nonzero coefficient but was not found!")
            else:
                self.cost_names.append(name)
                self.cost_functions.append(getattr(self.cost_container, '_cost_' + name))
        # cost episode sums
        self.cost_dict = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.cost_scales.keys()
        }

    def compute_reward(self):
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            term = self.reward_functions[i]()
            if name == "tracking_lin_vel":
                cmd_mag = torch.norm(self._commands[:, 0:2], dim=1)
                valid = cmd_mag > 0.1
                self._ep_track_sum += term * valid.float()
                self._ep_track_count += valid.float()
            rew = term * self.reward_scales[name]
            self.rew_buf += rew
            if torch.sum(rew) >= 0:
                self.rew_buf_pos += rew
            elif torch.sum(rew) <= 0:
                self.rew_buf_neg += rew
            self.episode_sums[name] += rew
        self.episode_sums["total"] += self.rew_buf
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self.reward_container._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        
    def compute_cost(self):
        costs = []
        for i in range(len(self.cost_functions)):
            name = self.cost_names[i]
            cost = self.cost_functions[i]() * self.cost_scales[name]
            self.cost_dict[name] = cost
            costs.append(cost)
        self.cost_buf = torch.stack(costs, dim=-1)

    def _resample_commands(self, env_ids):
        self._commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1],(len(env_ids), 1), device=self.device).squeeze(1)

        self._commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1],(len(env_ids), 1), device=self.device).squeeze(1)

        yaw_stop = (torch.rand(len(env_ids), device=self.device) < 0.30)
        self._yaw_stop_mask[env_ids] = yaw_stop

        if self.cfg.commands.heading_command:
            self._heading_cmd[env_ids] = torch_rand_float(
                self.command_ranges["heading"][0], self.command_ranges["heading"][1],(len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self._commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1],(len(env_ids), 1), device=self.device).squeeze(1)

    def compute_foot_height_data(
        self,
        foot_scanners,          # dict: {"hr","hl","fr","fl"}
        max_clearance=0.45,
        clip_range=(-1.0, 1.0),
    ):
        foot_clearance_list = []     # [N] × 4   (RAW meters)
        foot_height_vec_list = []    # [N, 9] × 4 (NORMALIZED)

        for key in ["fl", "fr", "rl", "rr"]:
            scanner = foot_scanners[key]

            ray_hits_z = scanner.data.ray_hits_w[..., 2]          # [N, 9]
            sensor_z = scanner.data.pos_w[:, 2].unsqueeze(1)      # [N, 1]

            valid = torch.isfinite(ray_hits_z)                    # [N, 9]

            # ---------- foot_height_vec (normalized) ----------
            # invalid는 "매우 멀다"로 취급해서 clamp(=1) 되게 유지 (기존 로직 유지)
            ground_z = torch.where(valid, ray_hits_z, torch.full_like(ray_hits_z, -1e6))
            clearance_vec = sensor_z - ground_z                   # [N, 9]
            clearance_vec_norm = torch.clamp(
                clearance_vec / max_clearance, clip_range[0], clip_range[1]
            )
            foot_height_vec_list.append(clearance_vec_norm)

            # ---------- foot_clearance (raw meters) ----------
            # max를 구할 때 invalid는 -inf로 빼고, 전부 invalid면 0으로 처리(원하시면 max_clearance로 변경 가능)
            ground_z_for_max = ray_hits_z.masked_fill(~valid, -torch.inf)
            ground_z_max = ground_z_for_max.max(dim=1).values     # [N]
            valid_any = valid.any(dim=1)                          # [N]

            clearance_raw = sensor_z[:, 0] - ground_z_max         # [N]  (meters)
            clearance_raw = torch.where(valid_any, clearance_raw, torch.zeros_like(clearance_raw))
            foot_clearance_list.append(clearance_raw)

        foot_clearance = torch.stack(foot_clearance_list, dim=1) - 0.035  # [N, 4]  RAW meters
        foot_height_vec = torch.cat(foot_height_vec_list, dim=1)  # [N, 36] normalized

        return foot_clearance, foot_height_vec

    def _update_lin_vel_filter(self):
        v_xy = self._robot.data.root_lin_vel_b[:, :2]  # (N,2)
        tau = 1.0  # 1초(=보행주기) 기준으로 저주파만 남김
        alpha = self.dt / (tau + self.dt)  # (0,1)
        self._lin_vel_xy_filt = (1.0 - alpha) * self._lin_vel_xy_filt + alpha * v_xy

    def _init_zmp_preview_buffers(self, gait_freq: float = 1.0, n_steps: int = 4):
        # fixed params
        self._zmp_gait_freq = float(gait_freq)
        self._zmp_n_steps   = int(n_steps)
        self._zmp_T_step    = 1.0 / (self._zmp_gait_freq * self._zmp_n_steps)
        self._zmp_wp_H  = self._zmp_n_steps + 1
        self._zmp_seg   = self._zmp_wp_H - 1
        N = int(round((self._zmp_seg * self._zmp_T_step) / self.dt))
        self._zmp_ref_N = max(N, 1)
        k = torch.arange(self._zmp_ref_N, device=self.device, dtype=torch.float32)          # [N]
        bin_idx = torch.floor((k * self.dt) / self._zmp_T_step).to(torch.int64)             # [N]
        self._zmp_hold_bin_idx = torch.clamp(bin_idx, 0, self._zmp_seg - 1)                 # [N]
        # buffers (root_xy로 full-snap)
        root_xy = self._robot.data.root_pos_w[:, :2].clone()                                # [B,2]
        self.zmp_wp_xy_seq  = root_xy[:, None, :].repeat(1, self._zmp_wp_H, 1)              # [B,H,2]
        self.zmp_ref_xy_seq = root_xy[:, None, :].repeat(1, self._zmp_ref_N, 1)             # [B,N,2]
        self.zmp_wp_xy      = root_xy.clone()                                               # [B,2]
        # zero enter detect
        self.prev_zero_command = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.com_ref_xy_seq  = torch.zeros(self.num_envs, self._zmp_ref_N, 2, device=self.device)
        self.com_ref_dxy_seq = torch.zeros(self.num_envs, self._zmp_ref_N, 2, device=self.device)

    def _update_zmp_preview_from_cmd(self, gait_prev: torch.Tensor, enter_zero: torch.Tensor):
        n_steps = self._zmp_n_steps

        bin_prev = torch.floor(gait_prev * n_steps).to(torch.int64)
        bin_now  = torch.floor(self.gait_indices * n_steps).to(torch.int64)
        step_event = (bin_now != bin_prev)

        # 1) enter_zero: full snap
        if enter_zero.any():
            root_xy = self._robot.data.root_pos_w[enter_zero, :2]
            self.zmp_wp_xy_seq[enter_zero]  = root_xy[:, None, :].repeat(1, self._zmp_wp_H, 1)
            self.zmp_ref_xy_seq[enter_zero] = root_xy[:, None, :].repeat(1, self._zmp_ref_N, 1)
            self.zmp_wp_xy[enter_zero]      = root_xy

        # 2) moving + step_event: shift + append
        update = step_event & (~self.zero_command)
        if update.any():
            v_b = self._commands[:, :2]
            v_w = quat_apply_yaw(
                self._robot.data.root_quat_w,
                torch.cat([v_b, torch.zeros_like(v_b[:, :1])], dim=-1),
            )[:, :2]

            seq_u = self.zmp_wp_xy_seq[update].clone()     # ✅ overlap 방지
            seq_u[:, :-1] = seq_u[:, 1:]
            seq_u[:, -1]  = seq_u[:, -2] + v_w[update] * self._zmp_T_step

            self.zmp_wp_xy_seq[update] = seq_u
            self.zmp_wp_xy[update]     = seq_u[:, 0]
            self.zmp_ref_xy_seq[update] = seq_u[:, self._zmp_hold_bin_idx, :]

    def _update_com_preview_from_zmp_ref(self, enter_zero: torch.Tensor):
        g = 9.81
        h = 0.5
        w = (g / h) ** 0.5
        dt = float(self.dt)
        c = float(np.cosh(w * dt))
        s = float(np.sinh(w * dt))
        B = self.num_envs
        N = self._zmp_ref_N

        # 1) enter_zero: com_ref full snap, vel=0
        if enter_zero.any():
            root_xy = self._robot.data.root_pos_w[enter_zero, :2]
            self.com_ref_xy_seq[enter_zero]  = root_xy[:, None, :].repeat(1, N, 1)
            self.com_ref_dxy_seq[enter_zero] = 0.0

        # 2) moving env만 forward sim으로 채움
        moving = ~self.zero_command
        if not moving.any():
            return

        x = self._robot.data.root_pos_w[:, :2]  # [B,2]
        v_b = self._robot.data.root_lin_vel_b[:, :2]
        xd = quat_apply_yaw(
            self._robot.data.root_quat_w,
            torch.cat([v_b, torch.zeros_like(v_b[:, :1])], dim=-1),
        )[:, :2]

        com_xy  = self.com_ref_xy_seq
        com_dxy = self.com_ref_dxy_seq

        for k in range(N):
            p = self.zmp_ref_xy_seq[:, k, :]  # [B,2]
            x0, xd0 = x, xd
            x  = c * x0 + (s / w) * xd0 + (1.0 - c) * p
            xd = (w * s) * x0 + c * xd0 - (w * s) * p

            com_xy[moving,  k, :] = x[moving]
            com_dxy[moving, k, :] = xd[moving]






