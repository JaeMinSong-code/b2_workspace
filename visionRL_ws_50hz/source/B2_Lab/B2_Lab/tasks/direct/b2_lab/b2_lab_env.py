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

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self._action_scale * self._actions + self._robot.data.default_joint_pos
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
        # 베이스 병진속도를 뺀 '몸통 대비 상대 발속도' (body frame).
        # 몸통이 전진할 때 정상 stance/swing 발이 페널티를 받지 않도록,
        # 발이 몸통 진행방향과 다르게 움직이는 성분만 남긴다.
        self.foot_rel_velocities_b = self.foot_velocities_b - self._robot.data.root_lin_vel_b.unsqueeze(1)  # [num_envs, 4, 3]

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
        )  # total = 49

        # 센서 노이즈는 history에 push되기 전에 적용 → CENet/actor가 noisy 입력으로 학습.
        # critic_obs와 CENet의 recon/vel target은 clean 유지 (privileged 정보)
        if self.cfg.add_observation_noise:
            noisy_obs = obs + torch.randn_like(obs) * self.obs_noise_std_vec
        else:
            noisy_obs = obs

        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([noisy_obs] * self.cfg.num_history_len, dim=1),
            torch.cat(
                [self.obs_history_buf[:, 1:], noisy_obs.unsqueeze(1)],
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

        self.obs_buf = noisy_obs
        self.critic_obs = torch.cat(
            (
                obs,  # critic은 clean obs 사용
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
            "prop_obs_clean": obs,  # CENet recon target용 (denoising 학습)
            "prop_obs_history": self.flattened_history_obs,
            "height_obs": height_data,
            "velocity_estimator_obs": self.flattened_history_obs,
            # critic의 lin_vel 스케일(x2.0)과 통일 — CENet v_hat도 같은 space가 됨
            "velocity_estimator_target": self._robot.data.root_lin_vel_b * 2.0,
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
        # died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return died, time_out

    def _update_terrain_levels(self, env_ids: torch.Tensor):
        """이번 에피소드 이동거리를 보고 해당 env의 terrain level을 올리거나 내린다.
        (IsaacLab ``mdp.terrain_levels_vel`` 과 동일 로직, direct env용 이식)
        주의: root state를 새 위치로 쓰기 전에(=에피소드 종료 위치일 때) 호출해야 한다."""
        # 스폰 원점에서 실제로 이동한 수평 거리
        distance = torch.norm(
            self._robot.data.root_pos_w[env_ids, :2] - self._terrain.env_origins[env_ids, :2],
            dim=1,
        )
        # 지형 한 칸의 절반 이상 이동 → 진급
        move_up = distance > self._terrain.cfg.terrain_generator.size[0] / 2
        # 명령대로 갔어야 할 거리의 절반도 못 감 → 강등 (진급 대상은 제외)
        move_down = (
            distance < torch.norm(self._commands[env_ids, :2], dim=1) * self._max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self._terrain.update_env_origins(env_ids, move_up, move_down)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        # terrain curriculum: root state를 덮어쓰기 전에 레벨 갱신.
        # plane 지형(terrain_generator=None)에서는 절대 실행 안 함(방어).
        if getattr(self.cfg, "terrain_curriculum", False) and self._terrain.cfg.terrain_generator is not None:
            self._update_terrain_levels(env_ids)
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
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
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

        # terrain curriculum 진행도 로깅 (평균 terrain level). plane 지형엔 terrain_levels 없음(방어).
        if getattr(self.cfg, "terrain_curriculum", False) and getattr(self._terrain, "terrain_levels", None) is not None:
            self.extras["log"]["Curriculum/terrain_level"] = torch.mean(self._terrain.terrain_levels.float())

        self._ep_track_sum[env_ids] = 0.0
        self._ep_track_count[env_ids] = 0.0

    def _step_contact_targets(self):
        # gait_prev = self.gait_indices.clone()
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * 1.4, 1.0)
        durations = torch.full((self.num_envs, 1), 0.6, device=self.device)
        trott = torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device).unsqueeze(0)
        gait_offsets = trott
        foot_indices = torch.remainder(self.gait_indices.unsqueeze(1) + gait_offsets, 1.0)
        self.foot_indices = foot_indices.clone()
        self.zero_command = (
            self._commands[:, :3].abs() < self.cfg.commands.zero_command_threshold
        ).all(dim=1)

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
        # stance 구간 [0, durations)를 smoothing한 목표접촉. durations를 반영해야
        # clock_inputs(관측)·desired_contact_states(목표)·swing mask가 같은 duty를 가리킴.
        # durations=0.5면 기존과 동일, 0.6이면 대각쌍 stance가 겹쳐 4점지지 목표가 생김.
        kappa = self.cfg.rewards.kappa_gait_probs
        normal_dist = torch.distributions.normal.Normal(0, kappa)
        r = torch.remainder(self.foot_indices, 1.0)
        smoothing = (normal_dist.cdf(r) * (1.0 - normal_dist.cdf(r - durations))
                     + normal_dist.cdf(r - 1.0) * (1.0 - normal_dist.cdf(r - (1.0 + durations))))
        self.desired_contact_states = smoothing

        # zero_command으로 래치된 발은 index=0(stance 경계, 목표접촉 0.5)이라
        # c5 gait_pattern 비용이 접촉 여부와 무관한 상수(0.5)가 되어 제약이 무력화됨.
        # 정지 시엔 4발 stance를 강제해야 하므로 래치된 발의 목표접촉을 1.0으로 덮어씀.
        self.desired_contact_states = torch.where(
            self.zero_hold_state,
            torch.ones_like(self.desired_contact_states),
            self.desired_contact_states,
        )

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

        # observation noise는 _get_observations 내부에서 history push 전에 적용됨
        self.obs_buf = self._get_observations()

        # RSL-RL extras
        if "observations" not in self.extras:
            self.extras.update(self.obs_buf)

        self._post_physics_step()
        return self.obs_buf, self.rew_buf, self.cost_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _parse_cfg(self, cfg):
        # control dt = sim dt × decimation (200Hz × 4 = 50Hz). 하드코딩 대신 유도해
        # sim/decimation 바뀌어도 reward scale(*=dt)·gait·episode length가 자동 정합.
        self.dt = self.cfg.sim.dt * self.cfg.decimation
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
        self._hip_roll_joint_ids, _ = self._robot.find_joints(".*_hip_joint")
        self._action_scale = torch.full((self.num_actions,), self.cfg.action_scale, device=self.device)
        self._action_scale[self._hip_roll_joint_ids] *= self.cfg.hip_roll_action_scale_factor
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._heading_cmd = torch.zeros(self.num_envs, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
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

        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies([".*calf.*"])
        self._body_contact_ids, _ = self._contact_sensor.find_bodies("base_link", ".*thigh.*")
        # 발 4개의 collision shape 인덱스를 body→shape 매핑으로 정확히 계산 (USD 마다 shape 개수 상이)
        self._foot_shape_ids = self._compute_foot_shape_ids()
        self.actuator_stiffness_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actuator_damping_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        mat = self._robot.root_physx_view.get_material_properties().to(self.device)  # [N, num_shapes, 3]
        self.dynamic_fric_coeffs = mat[:, self._foot_shape_ids, 1]
        # privileged obs용 gain 편차: 기본값 대비 상대 비율(±0.1 수준)로 저장.
        # actuator 그룹별로 해당 joint 인덱스에만 기록 (그룹이 여러 개여도 안전)
        for actuator in self._robot.actuators.values():
            ids = actuator.joint_indices
            self.actuator_stiffness_gains[:, ids] = (
                actuator.stiffness / self._robot.data.default_joint_stiffness[:, ids].clamp(min=1e-6) - 1.0
            )
            self.actuator_damping_gains[:, ids] = (
                actuator.damping / self._robot.data.default_joint_damping[:, ids].clamp(min=1e-6) - 1.0
            )

        # observation noise: 물리 단위 std에 obs scale을 곱해 scaled space 벡터로 변환.
        # cmd(6:9)/actions(33:45)/clock(45:49)은 내부 생성값이라 노이즈 0
        self.obs_noise_std_vec = torch.zeros(self.cfg.num_proprio, device=self.device)
        self.obs_noise_std_vec[0:3] = self.cfg.noise_std.ang_vel * 0.25
        self.obs_noise_std_vec[3:6] = self.cfg.noise_std.gravity
        self.obs_noise_std_vec[9:21] = self.cfg.noise_std.joint_pos
        self.obs_noise_std_vec[21:33] = self.cfg.noise_std.joint_vel * 0.05
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.rew_buf_pos = torch.zeros(self.num_envs, device=self.device)
        self.rew_buf_neg = torch.zeros(self.num_envs, device=self.device)
        self.cost_buf = torch.zeros(self.num_envs, device=self.device)
        self.obs_buf = {}  # observation 딕셔너리 초기화
        self.com_height = torch.zeros(self.num_envs, 1, device=self.device)

        # 주의: actor_obs는 항상 0 텐서. 값은 안 쓰이지만 obs dict "policy" 키의 shape을
        # runner가 num_actor_obs(=prop+v+z 차원) 결정에 사용하므로 지우면 안 됨.
        # 실제 actor 입력은 runner/alg에서 prop_obs + CENet(v_hat, z_hat)로 직접 조립됨.
        self.actor_obs = torch.zeros(self.num_envs, self.cfg.num_actor_obs, dtype=torch.float, device=self.device, requires_grad=False)
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
        self.foot_rel_velocities_b = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_joint_pos_target = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_joint_pos_target = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.foot_force = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.foot_contact_time = torch.zeros((self.num_envs, 4), device=self.device)
        self.zero_command = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 제로 커맨드 래치: 발별로 스윙 종료(stance 진입) 이후 1로 고정
        self.zero_hold_state = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        # episode tracking accumulators
        self._ep_track_sum = torch.zeros(self.num_envs, device=self.device)      # 누적 점수
        self._ep_track_count = torch.zeros(self.num_envs, device=self.device)    # 누적 스텝 수
        self.footscanner_height_data = torch.zeros(self.num_envs, 4, device=self.device)
        self.foot_height_vec = torch.zeros(self.num_envs, 4 * 25, device=self.device)
        self.commands_scale = torch.tensor([2.0, 2.0, 0.25], device=self.device, requires_grad=False,)

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
        from .b2quad_parkour_reward import B2quadParkourReward
        reward_containers = {"B2quadReward": B2quadReward, "B2quadParkourReward": B2quadParkourReward}
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

        if self.cfg.commands.heading_command:
            self._heading_cmd[env_ids] = torch_rand_float(
                self.command_ranges["heading"][0], self.command_ranges["heading"][1],(len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self._commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1],(len(env_ids), 1), device=self.device).squeeze(1)

        zero_command_mask = (
            torch.rand(len(env_ids), device=self.device) < self.cfg.commands.zero_command_probability
        )

        # 5% mask 밖에서 우연히 세 command가 모두 threshold 안에 들어오는 경우는
        # x command를 경계값으로 밀어 zero-command 판정에서 제외한다.
        accidental_zero_mask = (~zero_command_mask) & (
            self._commands[env_ids, :3].abs() < self.cfg.commands.zero_command_threshold
        ).all(dim=1)
        accidental_zero_env_ids = env_ids[accidental_zero_mask]
        if len(accidental_zero_env_ids) > 0:
            x_commands = self._commands[accidental_zero_env_ids, 0]
            x_signs = torch.where(x_commands >= 0.0, 1.0, -1.0)
            self._commands[accidental_zero_env_ids, 0] = (
                x_signs * self.cfg.commands.zero_command_threshold
            )

        self._commands[env_ids[zero_command_mask], :3] = 0.0

    def compute_foot_height_data(
        self,
        foot_scanners,          # dict: {"hr","hl","fr","fl"}
        max_clearance=0.45,
        clip_range=(-1.0, 1.0),
    ):
        foot_clearance_list = []     # [N] × 4 (RAW meters)
        foot_height_vec_list = []    # [N, num_rays] × 4 (NORMALIZED)

        for foot_idx, key in enumerate(["fl", "fr", "rl", "rr"]):
            scanner = foot_scanners[key]

            ray_hits_z = scanner.data.ray_hits_w[..., 2]          # [N, num_rays]
            # RayCaster origin은 발보다 0.30 m 위에 있으므로 sensor pos가 아니라
            # articulation의 실제 foot body 중심을 clearance 기준으로 사용한다.
            foot_z = self._robot.data.body_pos_w[
                :, self._feet_ids[foot_idx], 2
            ].unsqueeze(1)                                         # [N, 1]

            valid = torch.isfinite(ray_hits_z)                    # [N, num_rays]

            # ---------- foot_height_vec (normalized) ----------
            # invalid는 "매우 멀다"로 취급해서 clamp(=1) 되게 유지 (기존 로직 유지)
            ground_z = torch.where(valid, ray_hits_z, torch.full_like(ray_hits_z, -1e6))
            clearance_vec = foot_z - ground_z - self.cfg.foot_radius
            clearance_vec_norm = torch.clamp(
                clearance_vec / max_clearance, clip_range[0], clip_range[1]
            )
            foot_height_vec_list.append(clearance_vec_norm)

            # ---------- foot_clearance (raw meters) ----------
            # max를 구할 때 invalid는 -inf로 빼고, 전부 invalid면 0으로 처리(원하시면 max_clearance로 변경 가능)
            ground_z_for_max = ray_hits_z.masked_fill(~valid, -torch.inf)
            ground_z_max = ground_z_for_max.max(dim=1).values     # [N]
            valid_any = valid.any(dim=1)                          # [N]

            clearance_raw = foot_z[:, 0] - ground_z_max - self.cfg.foot_radius
            clearance_raw = torch.where(valid_any, clearance_raw, torch.zeros_like(clearance_raw))
            foot_clearance_list.append(clearance_raw)

        foot_clearance = torch.stack(foot_clearance_list, dim=1)  # [N, 4] raw meters
        foot_height_vec = torch.cat(foot_height_vec_list, dim=1)  # [N, 4*num_rays] normalized

        return foot_clearance, foot_height_vec
