import torch

class B2quadReward:
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    # ------------ reward functions ----------------

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.env._robot.data.root_lin_vel_b[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.env._robot.data.root_ang_vel_b[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.env._robot.data.projected_gravity_b[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target using COM height from height scanner
        target_height = self.env.cfg.target_height
        com_height = self.env.com_height.squeeze(-1)  # (num_envs,)
        height_error = torch.square(com_height - target_height)
        return height_error

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.env._robot.data.applied_torque), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.env._robot.data.joint_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.env._robot.data.joint_acc), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env._actions - self.env.last_actions), dim=1)

    def _reward_action_smoothness_1(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target - self.env.last_joint_pos_target)
        diff = diff * (self.env.last_actions[:, :24] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target - 2 * self.env.last_joint_pos_target + self.env.last_last_joint_pos_target)
        diff = diff * (self.env.last_actions[:, :24] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, :24] != 0)  # ignore second step
        return torch.sum(diff, dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.env.reset_terminated.float() * (~self.env.reset_time_outs.bool()).float()

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(
            torch.square(self.env._commands[:, :2] - self.env._robot.data.root_lin_vel_b[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma)

    def _reward_penalty_lin_vel(self):
        lin_vel_error = torch.sum(
            torch.square(self.env._commands[:, :2] - self.env._robot.data.root_lin_vel_b[:, :2]), dim=1
        )
        return lin_vel_error

    def _reward_tracking_foot_pos(self):
        lin_vel_error = torch.sum(
            torch.square(self.env._commands[:, :2] - self.env.foot_velocity_mean[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.env._commands[:, 2] - self.env._robot.data.root_ang_vel_b[:, 2])
        return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma)

    def _reward_penalty_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.env._commands[:, 2] - self.env._robot.data.root_ang_vel_b[:, 2])
        return ang_vel_error

    def _reward_feet_air_time(self):
        # feet air time
        first_contact = self.env._contact_sensor.compute_first_contact(self.env.step_dt)[:, self.env._feet_ids]
        last_air_time = self.env._contact_sensor.data.last_air_time[:, self.env._feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (~self.env.zero_command)
        return air_time

    def _reward_feet_clearance(self):
        feet_in_air = ~self.env.foot_contact_state

        # 발이 공중에 있을 때만 높이에 대한 리워드 적용
        feet_heights = self.env.foot_positions[:, :, 2]
        desired_clearance = 0.2  # 원하는 발 높이 (m)

        # 0.2m에 가까울수록 큰 리워드 (거리 기반 리워드)
        height_error = torch.norm(feet_heights - desired_clearance)
        # exp(-error)로 0.2에 가까울수록 1에 가까워짐
        clearance_reward_per_foot = torch.exp(-height_error / 0.05)  # 0.05는 scaling factor

        clearance_reward = torch.sum(clearance_reward_per_foot * feet_in_air.float(), dim=1) * (~self.env.zero_command)
        return torch.clamp(clearance_reward, min=0.0)  # 음수 방지

    def _reward_feet_clearance2(self):
        feet_in_air = ~self.env.foot_contact_state

        # 발이 공중에 있을 때만 높이에 대한 리워드 적용
        feet_heights = self.env.foot_positions[:, :, 2]
        desired_clearance = 0.2  # 원하는 발 높이 (m)

        # 0.2m에 가까울수록 큰 리워드 (거리 기반 리워드)
        height_error = torch.norm(feet_heights - desired_clearance)

        clearance_reward = torch.sum(height_error * feet_in_air.float(), dim=1) * (~self.env.zero_command)
        return torch.clamp(clearance_reward, min=0.0)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        current_joint_pos = self.env._robot.data.joint_pos
        default_joint_pos = self.env._robot.data.default_joint_pos

        # 관절 위치 편차 페널티
        joint_deviation = torch.sum(torch.abs(current_joint_pos - default_joint_pos), dim=1)

        return (joint_deviation) * self.env.zero_command

    def _reward_vel_stand_still(self):
        # Penalize motion at zero commands
        current_joint_vel = self.env._robot.data.joint_vel
        joint_vel_error = torch.norm(current_joint_vel, dim=-1).sum(dim=-1)
        all_feet_at_zero = torch.all(torch.abs(self.env.foot_indices) < 0.05, dim=1)
        return joint_vel_error * all_feet_at_zero.float()

    def _reward_joint_deviation_from_default(self):
        # 현재 관절 위치와 초기 관절 위치의 차이 계산
        current_joint_pos = self.env._robot.data.joint_pos
        default_joint_pos = self.env._robot.data.default_joint_pos
        joint_deviation = torch.sum(torch.square(current_joint_pos - default_joint_pos), dim=1)
        return joint_deviation

    def _reward_noslip(self):
        feet_slip = torch.sum(self.env.foot_contact_state * torch.norm(self.env.foot_force[:, :, :2], dim=-1), dim=-1) * (~self.env.zero_command)
        return feet_slip

    def _reward_no_slip_vel(self):
        foot_velocities = self.env._robot.data.body_lin_vel_w[:, self.env._feet_ids, :2]  # [N,4,2] 순서 [FL,FR,RL,RR]
        contact_mask = self.env.foot_contact_state  # [num_envs, 4] - True면 접촉
        foot_vel_magnitude = torch.norm(foot_velocities, dim=-1)  # [num_envs, 4]

        # 접촉된 발의 속도에 비례하여 패널티 적용
        slip_penalty = foot_vel_magnitude * contact_mask.float()  # 접촉된 발의 속도만큼 패널티
        return slip_penalty.sum(dim=1)

    def _reward_contact_vel(self):
        # ----- 지면(월드 z=0) 기준 가우시안 가중치 -----
        sigma = 0.025  # 5 cm 권장 (너무 작으면 가중치가 0으로 급락)
        delta = self.env.foot_positions[:, :, 2]                        # [B, L]
        contact_vel_weight = torch.exp(-(delta * delta) / (sigma * sigma))  # [B, L], 지면 가까울수록 1
        # ----- 스윙 페이즈 마스크 (접촉 X) -----
        is_swing = ~self.env.foot_contact_state                    # [B, L], True면 스윙
        # ----- 페널티 계산: 스윙 중, 지면 가까울수록, 발 속도가 빠를수록 크게 -----
        foot_speed = torch.linalg.norm(self.env.foot_velocities_b, dim=-1)   # [B, L]
        foot_contact_vel = torch.sum((foot_speed * contact_vel_weight) * is_swing, dim=1)  # [B]
        return foot_contact_vel
    
    def _reward_raibert_foot_placement(self):
        e = self.env
        swing = ~e.foot_contact_state                                  # [B,4] bool
        v, w = e._robot.data.root_lin_vel_b[:, :2], e._robot.data.root_ang_vel_b[:, 2]
        v_des, w_des = e._commands[:, :2], e._commands[:, 2]
        p, p0 = e.foot_pos_b[:, :, :2], e.default_foot_pos_b[:, :, :2]

        T = 1.0 / (1.0 * 4)  # gait_freq=1.0, n_steps=4  -> 0.25s
        k_ff, k_v, k_w, sig = 0.5, 0.2, 0.05, 0.15

        yaw_term = k_w * (w_des - w).view(-1, 1, 1) * torch.stack((-p0[..., 1], p0[..., 0]), dim=-1)
        p_des = p0 + (k_ff * T * v_des + k_v * (v_des - v))[:, None, :] + yaw_term

        r = torch.remainder(e.foot_indices, 1.0)
        gate = (swing & (((r - 0.7) / (0.3 + 1e-6)).clamp(0.0, 1.0) > 0.7)).float()  # stance_ratio=0.7, late_thr=0.7

        err2 = (p - p_des).pow(2).sum(dim=-1)
        per = torch.exp(-err2 / (sig * sig)) * gate
        return (per.sum(dim=1) / gate.sum(dim=1).clamp(min=1.0)) * (~e.zero_command).float()
    
    def _reward_support_polygon_center(self):
        foot_pos_xy = self.env.foot_pos_b[:, :, :2]  # [num_envs, 4, 2]
        contact_mask = self.env.foot_contact_state  # [num_envs, 4]
        # 3점 지지 상황 확인 (정확히 3개의 발이 접촉 중일 때만 계산)
        is_three_support = (torch.sum(contact_mask, dim=1) == 3)
        # 접촉한 발들의 위치 합 (접촉 안한 발은 마스킹되어 0이 됨)
        contact_pos_sum = torch.sum(foot_pos_xy * contact_mask.unsqueeze(-1).float(), dim=1)
        # 지지 다각형(삼각형)의 무게중심 (Centroid) 3점 지지 상황이므로 3으로 나눔
        centroid = contact_pos_sum / 3.0
        # Body frame 원점(0,0)과 무게중심 간의 거리 (XY 평면)Body frame 원점이 곧 로봇의 중심(COM)이라고 가정하고, 무게중심이 원점에 오도록 유도
        distance_sq = torch.sum(torch.square(centroid), dim=-1)
        # 거리가 가까울수록 보상 (Gaussian kernel)
        sigma = 0.05 
        reward = torch.exp(-distance_sq / (sigma ** 2))
        return reward * is_three_support.float()

    def _reward_tracking_lin_vel_LPF(self):
        v_xy = self.env._lin_vel_xy_filt  # 1초 LPF된 속도
        lin_vel_error = torch.sum(
            torch.square(self.env._commands[:, :2] - v_xy), dim=1
        )
        return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma)
    
    def _reward_avoid_stair_edge_from_footscan(self):
        # fh: normalized clearance = (sensor_z - ground_z) / max_clearance
        fh = self.env.foot_height_vec.view(-1, 4, 25)        # [N,4,25]
        fh = torch.clamp(fh, -0.95, 0.95)
        # 중앙 3x3 (5x5 ordering='xy' 가정)
        idx_3x3 = torch.tensor([6, 7, 8, 11, 12, 13, 16, 17, 18], device=fh.device, dtype=torch.long)
        fh9 = fh.index_select(dim=2, index=idx_3x3)          # [N,4,9]
        # "발 기준" reference: 중앙 9개 중 가장 작은 clearance (= 가장 높은 지면을 본 ray)
        ref = fh9.min(dim=2, keepdim=True).values            # [N,4,1]
        # 아래로 떨어진 정도만: clearance가 ref보다 커진 만큼 = 더 낮은 지면을 본 정도
        drop = fh9 - ref                                     # [N,4,9] (>=0이면 '아래')
        # 6 cm threshold
        max_clearance = 0.45
        thr = 0.06 / max_clearance                           # ≈ 0.13333 (normalized)
        # 6cm 이상 '아래'를 보는 ray 개수 (많을수록 패널티↑)
        bad = (drop > thr).float()                           # [N,4,9]
        bad_count = bad.sum(dim=2)                           # [N,4] 0..9
        contact = self.env.foot_contact_state.float()        # [N,4]
        # 옵션 A: 개수만 (요구사항 그대로)
        penalty = (bad_count / 9.0) * contact                # [N,4]
        # 옵션 B: 개수 + 깊이 (더 강하게/연속적으로)
        # penalty = (bad_count / 9.0) * excess * contact      # [N,4]
        return penalty.mean(dim=1)


