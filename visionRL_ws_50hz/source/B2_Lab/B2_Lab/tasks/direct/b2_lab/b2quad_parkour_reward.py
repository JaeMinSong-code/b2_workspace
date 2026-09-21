import torch

# =============================================================================
# Extreme Parkour reward container for the B2 IsaacLab direct env.
# Reference: Cheng et al., "Extreme Parkour with Legged Robots" (ICRA 2024).
#
# This mirrors the reward set + weights of the reference implementation, adapted
# to IsaacLab data APIs. It is a *separate* container from B2quadReward so the
# existing velocity-tracking pipeline is untouched. Select it by setting
#   cfg.reward_container_name = "B2quadParkourReward"
# and using cfg.rewards.scales from the parkour cfg block.
#
# Required env attributes (populated by the env in parkour_mode; see
# b2_lab_env.py `_update_parkour_goals` / `_init_parkour_buffers`):
#   env.goal_dir_b   : [N, 2]  unit goal direction in base-frame xy
#   env.cmd_vel      : [N]     commanded forward speed magnitude (m/s)
#   env.delta_yaw    : [N]     wrapped (target_yaw - base_yaw)
#   env._penalised_contact_ids : LongTensor of body ids for collision penalty
# Reference reward weights live in b2_lab_env_cfg.py (class rewards_parkour).
# =============================================================================


class B2quadParkourReward:
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    # ---- task rewards ------------------------------------------------------
    def _reward_tracking_goal_vel(self):
        # Reward velocity projected onto the goal direction, capped at cmd_vel.
        cur_vel = self.env._robot.data.root_lin_vel_b[:, :2]                # [N,2]
        goal_dir = self.env.goal_dir_b                                      # [N,2] unit
        cmd_vel = self.env.cmd_vel                                          # [N]
        proj = torch.sum(goal_dir * cur_vel, dim=1)                        # [N]
        return torch.minimum(proj, cmd_vel) / (cmd_vel + 1e-5)

    def _reward_tracking_yaw(self):
        # Encourage the base heading to track the goal direction.
        return torch.exp(-torch.abs(self.env.delta_yaw))

    # ---- base regularization ----------------------------------------------
    def _reward_lin_vel_z(self):
        return torch.square(self.env._robot.data.root_lin_vel_b[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.env._robot.data.root_ang_vel_b[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.env._robot.data.projected_gravity_b[:, :2]), dim=1)

    # ---- joint / actuation regularization ---------------------------------
    def _reward_dof_acc(self):
        return torch.sum(torch.square(self.env._robot.data.joint_acc), dim=1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.env._robot.data.applied_torque), dim=1)

    def _reward_delta_torques(self):
        last_torques = getattr(self.env, "last_torques", None)
        torques = self.env._robot.data.applied_torque
        if last_torques is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        return torch.sum(torch.square(torques - last_torques), dim=1)

    def _reward_action_rate(self):
        # EP uses an L2 norm (not sum-of-squares) of the action delta.
        return torch.norm(self.env.last_actions - self.env._actions, dim=1)

    def _reward_hip_pos(self):
        hip_ids = self.env._hip_roll_joint_ids
        cur = self.env._robot.data.joint_pos[:, hip_ids]
        default = self.env._robot.data.default_joint_pos[:, hip_ids]
        return torch.sum(torch.square(cur - default), dim=1)

    def _reward_dof_error(self):
        cur = self.env._robot.data.joint_pos
        default = self.env._robot.data.default_joint_pos
        return torch.sum(torch.square(cur - default), dim=1)

    # ---- contact-based penalties ------------------------------------------
    def _reward_collision(self):
        # Penalize contact forces on undesired bodies (thighs/calves/base).
        ids = getattr(self.env, "_penalised_contact_ids", None)
        if ids is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        net_forces = self.env._contact_sensor.data.net_forces_w             # [N, B, 3]
        contact = torch.norm(net_forces[:, ids], dim=-1) > 0.1              # [N, len(ids)]
        return torch.sum(contact.float(), dim=1)

    def _reward_feet_stumble(self):
        # Penalize large horizontal foot contact force relative to vertical.
        foot_force = self.env.foot_force                                    # [N, 4, 3]
        horiz = torch.norm(foot_force[:, :, :2], dim=-1)                   # [N, 4]
        vert = torch.abs(foot_force[:, :, 2])                             # [N, 4]
        stumble = torch.any(horiz > 4.0 * vert, dim=1)
        return stumble.float()

    def _reward_feet_edge(self):
        # Requires a terrain edge mask; returns zeros until the parkour terrain
        # exposes `env.feet_at_edge` ([N,4] bool). Keep scale in cfg so enabling
        # the terrain mask later activates this term with no code change.
        edge = getattr(self.env, "feet_at_edge", None)
        if edge is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        return torch.sum(edge.float(), dim=1)

    def _reward_termination(self):
        # Terminal penalty for non-timeout resets (falls / bad contacts).
        return self.env.reset_terminated.float() * (~self.env.reset_time_outs.bool()).float()
