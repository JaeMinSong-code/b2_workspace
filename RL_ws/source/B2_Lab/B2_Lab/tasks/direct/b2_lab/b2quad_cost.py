import torch
import torch.nn.functional as F


class B2quadCost:
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    # def _cost_c1com_height(self):
    #     com_height = self.env.com_height.squeeze(-1)  # (num_envs,)
    #     min_height = 0.2
    #     max_height = 0.42
    #     cost = torch.logical_or(com_height < min_height, com_height > max_height)
    #     return cost

    def _cost_c1com_height(self):
        # foot workspace (base frame) 경계, 발 순서 [FL, FR, RL, RR] — B2 기립자세 기준
        x_min = torch.tensor([0.10, 0.10, -0.60, -0.60], device=self.env.device).view(1, 4)
        x_max = torch.tensor([0.60, 0.60, -0.10, -0.10], device=self.env.device).view(1, 4)
        y_min = torch.tensor([0.05, -0.50, 0.05, -0.50], device=self.env.device).view(1, 4)
        y_max = torch.tensor([0.50, -0.05, 0.50, -0.05], device=self.env.device).view(1, 4)

        p = self.env.foot_pos_b  # [N, 4, 3] (body frame)
        violation_xy = (
            (p[..., 0] < x_min) | (p[..., 0] > x_max) | (p[..., 1] < y_min) | (p[..., 1] > y_max))  # [N, 4] bool
        feet_violate = violation_xy.any(dim=1)  # [N] bool

        # 2) CoM height constraint (B2 standing ~0.55)
        com_height = self.env.com_height.squeeze(-1)  # [N]
        min_height = 0.35
        max_height = 0.65
        com_violate = (com_height < min_height) | (com_height > max_height)  # [N] bool

        w_feet = 1.0
        w_com = 1.0
        cost = w_feet * feet_violate.float() + w_com * com_violate.float()  # [N] float

        return cost

    def _cost_c2dof_pos(self):
        joint_pos_limits = self.env._robot.data.joint_pos_limits
        scaled_upper_limits = joint_pos_limits[:, :, 1] * 0.98
        scaled_lower_limits = joint_pos_limits[:, :, 0] * 0.98
        upper_violation = (self.env._robot.data.joint_pos > scaled_upper_limits).clip(min=0.0)
        lower_violation = (self.env._robot.data.joint_pos < scaled_lower_limits).clip(min=0.0)

        # dynamic_violation = torch.norm(self.env._commands[:, :3], dim=1) * 0.5  # [num_envs]
        # dynamic_violation = dynamic_violation.unsqueeze(1)  # [num_envs, 1] - broadcasting을 위해
        # # 특정 DOF에 대해서만 동적 기준 적용
        # dof_ids = [4, 5, 6, 7]
        # upper_violation[:, dof_ids] = ((self.env._robot.data.joint_pos[:, dof_ids] > 0.05 ).float().clamp(min=0.))
        # lower_violation[:, dof_ids] = ((self.env._robot.data.joint_pos[:, dof_ids] < -0.05 ).float().clamp(min=0.))

        violations = torch.logical_or(lower_violation, upper_violation)
        violation_count = violations.sum(dim=1)
        cost = violation_count
        return cost

    def _cost_c3dof_vel(self):
        lower_violation = self.env._robot.data.joint_vel < -(10.0)
        upper_violation = self.env._robot.data.joint_vel > (10.0)
        violations = torch.logical_or(lower_violation, upper_violation)
        violation_count = violations.sum(dim=1)
        cost = violation_count / self.env.num_actions
        return cost

    def _cost_c4foot_clearance(self):
        durations = 0.7
        swing_start = durations

        swing_phase_mask = self.env.foot_indices > swing_start
        stance_phase_mask = self.env.foot_indices <= swing_start

        phases = torch.zeros_like(self.env.foot_indices)

        swing_progress = (self.env.foot_indices[swing_phase_mask] - swing_start) / (1.0 - swing_start)
        phases[swing_phase_mask] = torch.where(
            swing_progress <= 0.5,
            swing_progress * 2.0,
            2.0 - swing_progress * 2.0
        )
        phases[stance_phase_mask] = 0.0

        foot_swing_target = 0.15
        target_clearance = foot_swing_target * phases

        max_clearance_swing = 0.45
        max_clearance = torch.where(
            swing_phase_mask,
            torch.full_like(self.env.foot_indices, max_clearance_swing),
            torch.full_like(self.env.foot_indices, 0.03),
        )
        clearance = self.env.footscanner_height_data
        # foot_heights = self.env.foot_pos_b[:, :, 2] + self.env.com_height - 0.05
        swing_mask = self.env.desired_contact_states < -0.02
        low_swing_mask = torch.logical_and(swing_mask, clearance < target_clearance)
        low_swing_mask = clearance < target_clearance
        high_swing_mask = torch.logical_and(swing_mask, clearance > max_clearance)
        clearance_violation_mask = torch.logical_or(low_swing_mask, high_swing_mask)
        cost = clearance_violation_mask.float().sum(dim=1)
        return cost

    def _cost_c5gait_pattern(self):
        FL_phase = self.env.desired_contact_states[:, 0]
        FR_phase = self.env.desired_contact_states[:, 1]
        RL_phase = self.env.desired_contact_states[:, 2]
        RR_phase = self.env.desired_contact_states[:, 3]

        foot_contact_cost = (
            FL_phase * (1.0 - self.env.foot_contact_state[:, 0].float())
            + (1.0 - FL_phase) * self.env.foot_contact_state[:, 0].float()
        )
        foot_contact_cost += (
            FR_phase * (1.0 - self.env.foot_contact_state[:, 1].float())
            + (1.0 - FR_phase) * self.env.foot_contact_state[:, 1].float()
        )
        foot_contact_cost += (
            RL_phase * (1.0 - self.env.foot_contact_state[:, 2].float())
            + (1.0 - RL_phase) * self.env.foot_contact_state[:, 2].float()
        )
        foot_contact_cost += (
            RR_phase * (1.0 - self.env.foot_contact_state[:, 3].float())
            + (1.0 - RR_phase) * self.env.foot_contact_state[:, 3].float()
        )
        foot_contact_cost /= 4.0
        cost = foot_contact_cost
        return cost

    def _cost_torque_limit(self):
        torque_limit = 90.0

        applied_torques = self.env._robot.data.applied_torque  # [num_envs, num_joints]

        lower_violation = applied_torques < -torque_limit
        upper_violation = applied_torques > torque_limit
        violations = torch.logical_or(lower_violation, upper_violation)
        violation_count = violations.sum(dim=1)
        cost = violation_count / self.env.num_actions  # 16은 joint 개수
        return cost

    def _cost_c6undesired_contact(self):
        net_contact_forces = self.env._contact_sensor.data.net_forces_w_history
        contact_force = net_contact_forces[:, -1, self.env._undesired_contact_body_ids]  # [num_envs, num_bodies, 3]
        force_magnitudes = torch.norm(contact_force, dim=-1)  # [num_envs, num_bodies]
        max_force = torch.max(force_magnitudes, dim=1)[0]  # [num_envs]
        is_contact = max_force > 1.0  # [num_envs]
        cost = is_contact.float()  # [num_envs]
        cost * 100
        return cost

    def _cost_no_slip(self):
        foot_velocities = self.env._robot.data.body_lin_vel_w[:, self.env._feet_ids, :2]  # [N,4,2] 순서 [FL,FR,RL,RR]
        contact_mask = self.env.foot_contact_state  # [num_envs, 4] - True면 접촉
        foot_vel_magnitude = torch.norm(foot_velocities, dim=-1)  # [num_envs, 4]

        slip_threshold = 0.2

        # 연속적인 제약 위반 정도 계산
        slip_amount = torch.relu(foot_vel_magnitude - slip_threshold)
        cost = (slip_amount * contact_mask.float()).sum(dim=1)

        return cost

    def _cost_c4single_foot_lift_constraint(self):
        lift_min = 0.12
        lift_target = 0.20
        lift_max = 0.45
        stance_max = 0.03
        w_count = 1.0
        w_lift = 1.0
        w_stance = 1.0
        w_contact = 0.5
        # zero_command 전용 가중치(보통 강하게)
        w_zero_height = 2.0
        w_zero_contact = 2.0
        w_zero_liftany = 2.0  # "조금이라도 들면" 강벌점
        clearance = self.env.footscanner_height_data          # (N,4)
        contact = self.env.foot_contact_state.float()       # (N,4) 1=contact
        zero_cmd = self.env.zero_command                     # (N,) bool
        N, n_feet = clearance.shape

        # ======================
        # (A) move_cost: "딱 1개만 들어라" (기존 로직)
        # ======================
        lift_idx = clearance.argmax(dim=1)  # (N,)
        lift_oh = F.one_hot(lift_idx, num_classes=n_feet).to(clearance.dtype)  # (N,4)

        lift_h = (clearance * lift_oh).sum(dim=1)         # (N,)
        other_h = clearance * (1.0 - lift_oh)              # (N,4)

        lift_mask = (clearance > lift_min)                # (N,4)
        lift_count = lift_mask.sum(dim=1).float()          # (N,)
        count_cost = (lift_count - 1.0).pow(2)             # (N,)

        lift_low = F.relu(lift_target - lift_h)           # (N,)
        lift_high = F.relu(lift_h - lift_max)              # (N,)
        lift_cost = lift_low + lift_high                   # (N,)

        stance_cost = F.relu(other_h - stance_max).sum(dim=1)  # (N,)

        desired_contact = 1.0 - lift_oh
        contact_cost = (contact - desired_contact).abs().sum(dim=1) / n_feet  # (N,)

        move_cost = (
            w_count * count_cost +
            w_lift * lift_cost +
            w_stance * stance_cost +
            w_contact * contact_cost
        )

        # ======================
        # (B) stand_cost: zero_command일 때 "아무 발도 들지 마라"
        # ======================
        # 1) 모든 발이 stance_max 이하
        zero_height_cost = F.relu(clearance - stance_max).sum(dim=1)  # (N,)

        # 2) 모든 발 contact=1 (뜬 발/미끄러짐 등)
        zero_contact_cost = (1.0 - contact).sum(dim=1) / n_feet       # (N,)

        # 3) (선택) lift_min 초과인 발이 있으면 강벌점: "아예 들지마"
        zero_lift_any = (clearance > lift_min).sum(dim=1).float()     # (N,)
        zero_lift_any_cost = zero_lift_any.pow(2)                     # (N,)

        stand_cost = (
            w_zero_height * zero_height_cost +
            w_zero_contact * zero_contact_cost +
            w_zero_liftany * zero_lift_any_cost
        )

        # ======================
        # (C) switch: zero_command면 stand_cost, 아니면 move_cost
        # ======================
        cost = torch.where(zero_cmd, stand_cost, move_cost)
        return cost

