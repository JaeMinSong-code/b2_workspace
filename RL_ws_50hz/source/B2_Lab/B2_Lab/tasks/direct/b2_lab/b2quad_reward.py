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

    def _reward_roll_orientation(self):
        # 계단용: pitch(전후, idx 0)는 허용하고 roll(좌우, idx 1)만 벌 → 좌우 lean만 억제
        return torch.square(self.env._robot.data.projected_gravity_b[:, 1])

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
        first_contact = self.env._contact_sensor.compute_first_contact(self.env.step_dt)[:, self.env._feet_ids_contact]
        last_air_time = self.env._contact_sensor.data.last_air_time[:, self.env._feet_ids_contact]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (~self.env.zero_command)
        return air_time

    def _reward_feet_clearance(self):
        feet_in_air = ~self.env.foot_contact_state

        # 발이 공중에 있을 때만 높이에 대한 리워드 적용
        # 계단 지형 대응: 절대 world z가 아니라 ray 기반 지면-상대 clearance 사용
        feet_heights = self.env.footscanner_height_data
        desired_clearance = 0.2  # 원하는 발 높이 (m, 지면 대비)

        # 0.2m에 가까울수록 큰 리워드 (거리 기반 리워드)
        height_error = torch.norm(feet_heights - desired_clearance)
        # exp(-error)로 0.2에 가까울수록 1에 가까워짐
        clearance_reward_per_foot = torch.exp(-height_error / 0.05)  # 0.05는 scaling factor

        clearance_reward = torch.sum(clearance_reward_per_foot * feet_in_air.float(), dim=1) * (~self.env.zero_command)
        return torch.clamp(clearance_reward, min=0.0)  # 음수 방지

    def _reward_feet_clearance2(self):
        feet_in_air = ~self.env.foot_contact_state

        # 발이 공중에 있을 때만 높이에 대한 리워드 적용
        # 계단 지형 대응: 절대 world z가 아니라 ray 기반 지면-상대 clearance 사용
        feet_heights = self.env.footscanner_height_data
        desired_clearance = 0.2  # 원하는 발 높이 (m, 지면 대비)

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
        # 지면 '상대' clearance 기준 가우시안 가중치 (계단에서도 0=착지 기준이 맞음).
        # footscanner_height_data: [B,4] 각 발 밑 지면 대비 clearance(m), 순서 = foot_velocities_b 와 동일.
        foot_heights = self.env.footscanner_height_data                     # [B, 4]
        contact_vel_weight = torch.exp(-torch.square(foot_heights) / (0.01 ** 2))  # 지면 ~1cm 이내에서만 활성
        # 발이 지면 가까울수록 & 빠를수록 페널티 (swing/stance 라벨 무관, 높이 게이트가 대신함)
        foot_speed = torch.linalg.norm(self.env.foot_velocities_b, dim=-1)  # [B, 4]
        contact_vel_per_foot = foot_speed * contact_vel_weight             # [B, 4]
        # 평균 + 최악 발 강조
        foot_contact_vel = contact_vel_per_foot.mean(dim=1) + 0.5 * contact_vel_per_foot.max(dim=1).values
        # 리셋 직후 finite-diff 발속도 스파이크 제거 (prev_foot_positions=0 → 가짜 대속도)
        valid_step = (self.env.episode_length_buf > 1) & (~self.env.reset_buf.bool())
        return foot_contact_vel * valid_step.float()

    def _reward_swing_horizontal(self):
        # style: 지면 근처에선 수평(xy) 발 움직임 억제, 높이 뜰수록 게이트가 풀려 수평 이동 자유.
        # → 발이 수직으로 뜨고 내려오는 자연스러운 스윙 유도(계단 라이저 stubbing 감소).
        foot_heights = self.env.footscanner_height_data                     # [B, 4] 지면 대비 clearance(m)
        near_ground = torch.exp(-torch.square(foot_heights) / (0.06 ** 2))  # 지면 근처 1, ~10cm 이상이면 ~0
        # 수직(z) 이지/착지는 허용, 수평(xy) 성분만 페널티. foot_velocities_b는 planted foot이면 ≈0.
        horiz_speed = torch.linalg.norm(self.env.foot_velocities_b[:, :, :2], dim=-1)  # [B, 4]
        penalty_per_foot = horiz_speed * near_ground                        # [B, 4]
        style = penalty_per_foot.mean(dim=1) + 0.5 * penalty_per_foot.max(dim=1).values
        # contact_vel과 동일하게 리셋 직후 발속도 스파이크 제거
        valid_step = (self.env.episode_length_buf > 1) & (~self.env.reset_buf.bool())
        return style * valid_step.float()
