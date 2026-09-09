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
        # clearance 제약은 contact보다 보수적으로: c5/clock이 만드는 스윙(r>0.5)의
        # '중간 0.75 기준 ±35%(=중앙 70%)' 구간에서만 최소높이를 요구한다.
        # 스윙 가장자리(이지·착지)는 발이 낮게 있어야 하므로 최소·stance캡 둘 다 미적용.
        r = self.env.foot_indices                      # [N,4], 0~1
        stance_dur = 0.5                               # c5/clock durations 와 동일 (스윙 = r > 0.5)

        swing_mid = (stance_dur + 1.0) / 2.0           # 0.75  스윙 중간
        swing_half = (1.0 - stance_dur) / 2.0          # 0.25  스윙 반폭
        active_half = 0.70 * swing_half                # 중앙 70% → ±0.175 → [0.575, 0.925]

        dist = (r - swing_mid).abs()
        active_mask = dist <= active_half              # 중앙 70%에서만 최소높이 요구
        phases = torch.clamp(1.0 - dist / active_half, min=0.0)  # 0.75에서 1, 창 경계에서 0

        foot_swing_target = 0.10  # 스윙 중간 최소 clearance(하한). B2 평지 보행 기준 ~0.08~0.12m
        target_clearance = foot_swing_target * phases

        max_clearance_swing = 0.45
        max_clearance_stance = 0.03
        clearance = self.env.footscanner_height_data

        stance_mask = r < stance_dur                   # 진짜 stance만 (스윙 가장자리 제외)
        low_swing_mask = active_mask & (clearance < target_clearance)
        high_swing_mask = active_mask & (clearance > max_clearance_swing)
        high_stance_mask = stance_mask & (clearance > max_clearance_stance)
        clearance_violation_mask = low_swing_mask | high_swing_mask | high_stance_mask
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
