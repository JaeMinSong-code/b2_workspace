# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
from dataclasses import dataclass
from typing import Union
from tensordict import TensorDict


import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply, quat_apply_inverse, wrap_to_pi, quat_apply_yaw, euler_xyz_from_quat, yaw_quat

from .rbq_multi_critic_env_cfg import RBQEnvCfg

from isaaclab.managers import EventManager, CurriculumManager, CommandManager



class RBQEnv(DirectRLEnv):
    cfg: RBQEnvCfg
    def __init__(self, cfg: RBQEnvCfg , render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)

        self.original_env_id = 0
        self.sampling_envs_ids = torch.tensor(list(range(1, 10)), device=self.device)


        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "track_base_height_exp",
                "track_foot_height_exp",
                "track_gait_frequency",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "root_com_acc_l2",
                "dof_acc_l2",
                "feet_vel_l2",
                # "feet_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "feet_nominal_pos_l2",
                "feet_slip_l2",
                "undesired_contacts",
                "calf_contacts",
                "orientation_l2",
                "stand_orientation_PBRS",
                "stand_still",
                "symmetry",
                "contact_pattern",
                "feet_stumble_force",
                "termination",
                "joint_vel_penalty",
                "joint_limits",
                "joint_torques_log_barrier",
                "pitch_flip_rate",
                "alpha",
                "beta",
                "zrecoverability",
                "zquad_weight",
                "zbiped_weight",
            ]
        }
        # Get specific body indices
        print(f"self._bodies_name = {self._robot.data.body_names}")
        print(f"Contact sensor bodies_name = {self._contact_sensor.body_names}")
        ## !!caution!! not to confused with robot body order
        self._base_contact_id, _ = self._contact_sensor.find_bodies("base")
        print(f"self._base_contact_id = {self._base_contact_id}")
        ## !!caution!! not to confused with contact sensor body order
        self._base_body_id, _ = self._robot.find_bodies("base")
        print(f"self._base_body_id = {self._base_body_id}")

        self._fuselage_body_id, _ = self._robot.find_bodies("FL_hip")
        print(f"self._fuselage_body_id = {self._fuselage_body_id}")

        self._terminate_contact_body_ids, _ = self._contact_sensor.find_bodies(self.cfg.termination_contact_body_ids_list)
        print(f"self._terminate_contact_body_ids = {self._terminate_contact_body_ids}")

        self._feet_contact_ids, _ = self._contact_sensor.find_bodies(".*foot")
        print(f"self._feet_contact_ids = {self._feet_contact_ids}")
        self._feet_body_ids, _ = self._robot.find_bodies(".*foot")
        print(f"self._feet_body_ids = {self._feet_body_ids}")

        self._hip_joint_ids, _ = self._robot.find_joints(".*hip_joint")
        print(f"self._hip_joint_ids = {self._hip_joint_ids}")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(self.cfg.undesired_contact_body_ids_list)
        print(f"self._undesired_contact_body_ids = {self._undesired_contact_body_ids}")
        self._calf_contact_ids, _ = self._contact_sensor.find_bodies(self.cfg.calf_contact_body_ids_list)
        print(f"self._calf_contact_ids = {self._calf_contact_ids}")

        if self.cfg.events:
            self.event_manager = EventManager(self.cfg.events, self)
            print("[INFO] Event Manager: ", self.event_manager) 


        if self.cfg.events:
            if "startup" in self.event_manager.available_modes:
                self.event_manager.apply(mode="startup")


        self.heading_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.forward_vec = torch.tensor([1., 0., 0.], device=self.device).repeat((self.num_envs,1))
        # self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.history_length, self.cfg.prop_observation_space, dtype=torch.float, device=self.device) 

        self.speed_factor = torch.ones(self.num_envs, device=self.device) * self.cfg.init_speed_factor #0.3
        self.target_min_foot_height = torch.ones(self.num_envs, device=self.device) * (self.cfg.init_target_min_foot_height)
        
        self.placeholder16 = torch.zeros(self.num_envs, 16, device=self.device)

        self.foot_pos_w = self._robot.data.body_com_pos_w[:,self._feet_body_ids,:] - (self._robot.data.body_com_pos_w[:,self._base_body_id,:])
        self.foot_pos_b = torch.zeros_like(self.foot_pos_w)

        self.foot_pos_b[:, :, :] = quat_apply_inverse(self._robot.data.root_link_quat_w.repeat(1,4,1), self.foot_pos_w[:, :, :])

        self.default_feet_pose = torch.tensor([
            [ 0.29,  0.17], 
            [ 0.29, -0.17], 
            [-0.29,  0.17], 
            [-0.29, -0.17]], device="cuda:0")
        
        self.last_actions = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.last_last_last_actions = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        
        self.prev_stand_orientation_phi = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )


        self.last_foot_contact = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                            requires_grad=False)
        
        self.peak_foot_heights = torch.zeros(self.num_envs, 4,device=self.device,
                            requires_grad=False)
        self.peak_body_height = torch.zeros(self.num_envs, 4,device=self.device,
                        requires_grad=False)
        self.last_peak_foot_heights = torch.zeros(self.num_envs, 4,device=self.device,
                            requires_grad=False)
        self.peak_foot_vels = torch.zeros(self.num_envs, 4,device=self.device,
                            requires_grad=False)
        self.last_peak_foot_vels = torch.zeros(self.num_envs, 4,device=self.device,
                            requires_grad=False)
        self.current_foot_heights = torch.zeros(self.num_envs, 4,device=self.device,
                            requires_grad=False) 
        self.last_was_contact = torch.zeros(self.num_envs, 4, device=self.device)

        self.alpha =  torch.zeros(self.num_envs, device=self.device)
        self.beta =  torch.zeros(self.num_envs, device=self.device)
        self.is_standing = torch.zeros(self.num_envs, device=self.device)

        self.flip_phase = torch.zeros(self.num_envs, device=self.device)
        self.target_gravity = torch.tensor([1., 0., .0], device=self.device).repeat((self.num_envs, 1))

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        if self.cfg.terrain_curr_flag:
            self._height_scanner_dense = RayCaster(self.cfg.height_scanner_dense_cfg)
            self.scene.sensors["height_scanner"] = self._height_scanner_dense
            self._height_scanner_sparse = RayCaster(self.cfg.height_scanner_sparse_cfg)
            self.scene.sensors["height_scanner_far_sparse"] = self._height_scanner_sparse

        self._foot_scanner_RR = RayCaster(self.cfg.foot_scanner_RR)
        self.scene.sensors["foot_scanner_RR"] = self._foot_scanner_RR
        self._foot_scanner_RL = RayCaster(self.cfg.foot_scanner_RL)
        self.scene.sensors["foot_scanner_RL"] = self._foot_scanner_RL
        self._foot_scanner_FR = RayCaster(self.cfg.foot_scanner_FR)
        self.scene.sensors["foot_scanner_FR"] = self._foot_scanner_FR
        self._foot_scanner_FL = RayCaster(self.cfg.foot_scanner_FL)
        self.scene.sensors["foot_scanner_FL"] = self._foot_scanner_FL
        
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene._terrain = self._terrain
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # self.scene.filter_collisions()
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = (actions.clone()).clamp(-100, 100)

        # if self.common_step_counter % 960 == 0:
        #     self.alpha += 0.05
        #     if self.alpha > 1.0:
        #         self.alpha = 1.0

        # self.alpha = torch.ones(self.num_envs, device=self.device) * 0.5
        # self.is_standing = torch.ones(self.num_envs, device=self.device) * 1.0

        self.alpha = ((-self._robot.data.projected_gravity_b[:, 2]) / 1.0).clamp(0.0, 1.0)
        self.beta = (((self._robot.data.projected_gravity_b[:, 0])-0.0) / 1.0).clamp(0.0, 1.0) * self.is_standing 
        # stand_joint_pos = torch.tensor([
        #     0.0, 0.0, 0.0, 0.0, #HR
        #     0.6, 0.6, 0.6, 0.6, #HP
        #     -2.5, -2.5, -1.1, -1.1, #KP
        # ], device=self.device)

        
        stand_joint_pos = self._robot.data.default_joint_pos.clone()

        idx = torch.tensor([2, 3, 6, 7, 10, 11], device=stand_joint_pos.device)
        stand_joint_pos[:, idx] = self._robot.data.joint_pos[:, idx]
                
        
        quad_joint_pos= torch.tensor([
            0.0, 0.0, 0.0, 0.0, #HR
            0.7, 0.7, 0.7, 0.7, #HP
            -1.4, -1.4, -1.4, -1.4, #KP
        ], device=self.device)

        # self._processed_actions = self.cfg.action_scale * self._actions[:, :12] + ((self._robot.data.default_joint_pos * self.alpha) + (self._robot.data.joint_pos * (1.0 - self.alpha)))

        self._processed_actions = ((self.cfg.action_scale * self._actions[:, :12]) + 
                                   (self.alpha.unsqueeze(-1) * 1.0 * (quad_joint_pos - self._robot.data.joint_pos)) + 
                                   (self.beta.unsqueeze(-1) * 2.0 * (stand_joint_pos - self._robot.data.joint_pos)) + 
                                   self._robot.data.joint_pos)
        # self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.joint_pos


    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        forces = self._contact_sensor.data.net_forces_w_history[:, :, :, :]  # (N,H,feet,3)
        force_mag = torch.norm(forces, dim=-1)                                                    # (N,H,feet)
        is_contact = (force_mag.max(dim=1).values > 1.0)                                          # (N,feet)
        is_swing = ~is_contact

        contact_mask = is_contact[:, self._feet_contact_ids] # [N_env, N_foot] 
        body_heights = (-self.foot_pos_b[:, :, 2])
        body_heights_sum = torch.sum(body_heights * contact_mask, dim=1)
        contact_count = torch.sum(contact_mask, dim=1) # [N_env] 
        contact_count = torch.clamp(contact_count, min=1.0) # safety
        body_height = body_heights_sum / contact_count

        swing_mask = ~contact_mask # [N_env, N_foot] 
        foot_heights = (-self.foot_pos_b[:, :, 2])
        foot_heights_sum = torch.sum(foot_heights * swing_mask, dim=1)
        swing_count = torch.sum(swing_mask, dim=1) # [N_env] 
        swing_count = torch.clamp(swing_count, min=1.0) # safety
        foot_height = (foot_heights_sum / swing_count)

        

        if self.cfg.terrain_curr_flag:
            height_data_dense = (
            self._height_scanner_dense.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner_dense.data.ray_hits_w[..., 2] - 0.4
            ).clip(-5.0, 5.0)
            height_data_sparse = (
            self._height_scanner_sparse.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner_sparse.data.ray_hits_w[..., 2] - 0.4
            ).clip(-5.0, 5.0)

            nan_or_inf_mask = torch.isnan(height_data_dense) | torch.isinf(height_data_dense)
            if nan_or_inf_mask.any():
                height_data_dense[nan_or_inf_mask] = 0.0
            nan_or_inf_mask = torch.isnan(height_data_sparse) | torch.isinf(height_data_sparse)
            if nan_or_inf_mask.any():
                height_data_sparse[nan_or_inf_mask] = 0.0
        else:
            height_data_dense = None
            height_data_sparse = None


        # T = 2.0

        # # phase update
        # self.flip_phase = (self.flip_phase + self.step_dt / T) % 1.0

        # theta = 2.0 * torch.pi * self.flip_phase

        # # target gravity
        # self.target_gravity[:, 0] = torch.sin(theta)
        # self.target_gravity[:, 1] = 0.0
        # self.target_gravity[:, 2] = -torch.cos(theta)

        # self.target_normal[:, 0] = torch.cos(theta)
        # self.target_normal[:, 1] = 0.0
        # self.target_normal[:, 2] = -torch.sin(theta)

        commands_obs = torch.cat(
            [        
                self.commands[:, :2] * 2.0,
                self.commands[:, 2].unsqueeze(-1)* 0.25,
                # self.cfg.command_noise_cfg.func(self.commands[:, :2], self.cfg.command_noise_cfg) * 2.0,
                # self.cfg.command_noise_cfg.func(self.commands[:, 2].unsqueeze(-1), self.cfg.command_noise_cfg) * 0.25,
            ],
            dim=-1
        )


        prop_obs = torch.cat(
            [
                tensor
                for tensor in (
                    self.cfg.ang_vel_noise_cfg.func(self._robot.data.root_ang_vel_b, self.cfg.ang_vel_noise_cfg) * 0.25,
                    self.cfg.projected_gravity_noise_cfg.func(self._robot.data.projected_gravity_b, self.cfg.projected_gravity_noise_cfg) * 1.0,
                    (self.cfg.joint_pos_noise_cfg.func(self._robot.data.joint_pos, self.cfg.joint_pos_noise_cfg) - self._robot.data.default_joint_pos) * 1.0,
                    self.cfg.joint_vel_noise_cfg.func(self._robot.data.joint_vel, self.cfg.joint_vel_noise_cfg) * 0.05,
                    self.last_actions,
                    self.last_last_actions,
                    self.last_last_last_actions,
                    self.alpha.unsqueeze(-1),
                    self.beta.unsqueeze(-1),
                    self.is_standing.unsqueeze(-1),
                )
                if tensor is not None

            ],
            dim=-1,
        )

        actor_obs = torch.cat(
            [        
                commands_obs,
                prop_obs,
            ],
            dim=-1
        )

        current_air_time = self._contact_sensor.data.last_air_time[:, self._feet_contact_ids]
        current_contact_time = self._contact_sensor.data.last_contact_time[:, self._feet_contact_ids]

        critic_obs = torch.cat(
            [
                tensor
                for tensor in (
                    commands_obs,
                    self._robot.data.root_lin_vel_b *2.0,
                    self._robot.data.root_ang_vel_b *0.25,
                    self._robot.data.projected_gravity_b,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel *0.05,
                    self.last_actions,
                    self.last_last_actions,
                    self.last_last_last_actions,
                    height_data_dense,
                    height_data_sparse,
                    foot_heights,
                    is_contact[:, self._terminate_contact_body_ids],
                    is_contact[:, self._undesired_contact_body_ids],
                    is_contact[:, self._calf_contact_ids],
                    self._contact_sensor.data.last_air_time,
                    self._contact_sensor.data.last_contact_time,
                    self.peak_foot_heights,
                    current_air_time,
                    current_contact_time,
                    self.alpha.unsqueeze(-1),
                    self.beta.unsqueeze(-1),
                    self.is_standing.unsqueeze(-1),
                    # theta.unsqueeze(-1),
                )
                if tensor is not None
            ],
            dim=-1,
        )

        actor_obs = torch.nan_to_num(actor_obs, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-100, 100)
        critic_obs = torch.nan_to_num(critic_obs, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-100, 100)

        observations = {"policy": actor_obs,
                        "critic": critic_obs,
                        }
        return observations
    
    def _update_deviations(self):
        self.last_last_last_actions[:] = self.last_last_actions[:].clone().detach()
        self.last_last_actions[:] = self.last_actions[:].clone().detach()
        self.last_actions[:] = self._actions.clone().detach()


    def _update_peak_foot_height_buffer(self):
            #this function should be called before feet height cost calculation
        self.foot_pos_w = self._robot.data.body_link_pos_w[:,self._feet_body_ids,:] - (self._robot.data.root_link_pos_w[:, :]).unsqueeze(1)
        self.foot_pos_b[:, :, :] = quat_apply_inverse(self._robot.data.root_link_quat_w.repeat(1,4,1), self.foot_pos_w[:, :, :])
        # print(self.foot_pos_b[0, :, :2])
        foot_heights = (self.cfg.target_base_height +  self.foot_pos_b[:, :, 2].squeeze(-1))-0.026
        foot_heights = self._robot.data.body_link_pos_w[:, self._feet_body_ids, 2].squeeze(-1).clamp(min=0.0) -0.026

        foot_heights = torch.stack(
                [
                ((self._foot_scanner_FL.data.pos_w[:, 2]) - (self._foot_scanner_FL.data.ray_hits_w[:, :, 2].squeeze(1)))-0.04,
                ((self._foot_scanner_FR.data.pos_w[:, 2]) - (self._foot_scanner_FR.data.ray_hits_w[:, :, 2].squeeze(1)))-0.04,
                ((self._foot_scanner_RL.data.pos_w[:, 2]) - (self._foot_scanner_RL.data.ray_hits_w[:, :, 2].squeeze(1)))-0.04,
                ((self._foot_scanner_RR.data.pos_w[:, 2]) - (self._foot_scanner_RR.data.ray_hits_w[:, :, 2].squeeze(1)))-0.04,
                ],dim=-1).clamp(min=-5.0, max=5.0)
        nan_or_inf_mask = torch.isnan(foot_heights) | torch.isinf(foot_heights)
        if nan_or_inf_mask.any():
            foot_heights[nan_or_inf_mask] = 0.0


        forces = self._contact_sensor.data.net_forces_w_history[:, :, self._feet_contact_ids, :]  # (N,H,feet,3)
        force_mag = torch.norm(forces, dim=-1)                                                    # (N,H,feet)
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_contact_ids]

        # 1) 현재 시점 contact만 사용 (history max 지양)
        # is_contact = (force_mag[:, -1, :] > 1.0)   # (N,feet) bool
        is_contact = first_contact
        is_swing   = ~is_contact

        # 2) 이벤트 검출 (last_is_contact는 이전 step의 bool이어야 함)
        liftoff   = torch.logical_and(self.last_was_contact, is_swing)        # stance -> swing
        touchdown = torch.logical_and(torch.logical_not(self.last_was_contact), is_contact)  # swing -> stance

        # 3) liftoff에서 swing-peak 버퍼 리셋
        self.last_peak_foot_heights = torch.where(
            liftoff,
            torch.zeros_like(self.last_peak_foot_heights),
            self.last_peak_foot_heights
        )
        self.peak_body_height = torch.where(
            liftoff,
            torch.zeros_like(self.peak_body_height),
            self.peak_body_height
        )


        # 4) swing 동안 max foot height 추적
        self.last_peak_foot_heights = torch.where(
            is_swing & (foot_heights > self.last_peak_foot_heights),
            foot_heights,
            self.last_peak_foot_heights
        )

        # 5) touchdown 순간에 peak 확정 저장
        self.peak_foot_heights = torch.where(
            touchdown,
            self.last_peak_foot_heights,
            self.peak_foot_heights
        )
        self.peak_body_height= torch.where(
            touchdown,
            -self.foot_pos_b[:, :, 2],
            self.peak_body_height
        )


        # self.last_peak_foot_heights = torch.where(
        #     is_contact,
        #     torch.zeros_like(self.last_peak_foot_heights),
        #     self.last_peak_foot_heights
        # )
        
        self.current_foot_heights = foot_heights
        self.last_was_contact = is_contact
        # print(self.current_foot_heights[0])

    def joint_limit_log_barrier(
            self,
            q: torch.Tensor,
            q_min: torch.Tensor,
            q_max: torch.Tensor,
            margin_ratio: Union[float, torch.Tensor] = 0.1,
            eps: float = 1e-6,
        ) -> torch.Tensor:
            """
            q:            (num_envs, num_joints)
            q_min:        (num_joints,) or (num_envs, num_joints)
            q_max:        (num_joints,) or (num_envs, num_joints)
            margin_ratio: scalar, (num_joints,), or (num_envs, num_joints)

            returns:
                penalty_per_joint: (num_envs, num_joints)
            """
            # Ensure tensor conversion on correct device/dtype
            if not torch.is_tensor(margin_ratio):
                margin_ratio = torch.tensor(margin_ratio, device=q.device, dtype=q.dtype)
            else:
                margin_ratio = margin_ratio.to(device=q.device, dtype=q.dtype)

            q_min = q_min.to(device=q.device, dtype=q.dtype)
            q_max = q_max.to(device=q.device, dtype=q.dtype)

            joint_range = q_max - q_min
            margin = joint_range * margin_ratio

            # optional safety: prevent zero or negative margin
            margin = torch.clamp(margin, min=eps)

            dist_low = q - q_min
            dist_high = q_max - q

            x_low = dist_low / margin
            x_high = dist_high / margin

            low_barrier = -torch.log(torch.clamp(x_low, min=eps))
            high_barrier = -torch.log(torch.clamp(x_high, min=eps))

            low_barrier = torch.where(x_low < 1.0, low_barrier, torch.zeros_like(low_barrier))
            high_barrier = torch.where(x_high < 1.0, high_barrier, torch.zeros_like(high_barrier))

            penalty = low_barrier + high_barrier
            return penalty



    def _stand_orientation_phi(self):
        stand_orientation_error = (1.0 - self._robot.data.projected_gravity_b[:, 0]) ** 2
        stand_orientation_phi = torch.exp(-stand_orientation_error / 0.25)
        return stand_orientation_phi

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        self._update_peak_foot_height_buffer()

        if self.cfg.terrain_curr_flag:
            near_base_height = (
            self._height_scanner_dense.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner_dense.data.ray_hits_w[..., 2]
            ).clip(-5.0, 5.0)
            nan_or_inf_mask = torch.isnan(near_base_height) | torch.isinf(near_base_height)
            if nan_or_inf_mask.any():
                near_base_height[nan_or_inf_mask] = 0.0
        else:
            height_data_dense = None

        foot_heights = torch.stack(
        [
        ((self._foot_scanner_FL.data.pos_w[:, 2]) - (self._foot_scanner_FL.data.ray_hits_w[:, :, 2].squeeze(1)))-0.025,
        ((self._foot_scanner_FR.data.pos_w[:, 2]) - (self._foot_scanner_FR.data.ray_hits_w[:, :, 2].squeeze(1)))-0.025,
        ((self._foot_scanner_RL.data.pos_w[:, 2]) - (self._foot_scanner_RL.data.ray_hits_w[:, :, 2].squeeze(1)))-0.025,
        ((self._foot_scanner_RR.data.pos_w[:, 2]) - (self._foot_scanner_RR.data.ray_hits_w[:, :, 2].squeeze(1)))-0.025,
        ],dim=-1).clamp(min=-5.0, max=5.0)
        nan_or_inf_mask = torch.isnan(foot_heights) | torch.isinf(foot_heights)
        if nan_or_inf_mask.any():
            foot_heights[nan_or_inf_mask] = 0.0


        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0] > 1.0

        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_contact_ids]
        is_swing = ~is_contact

        is_commanding = torch.logical_or((torch.norm(self.commands[:, :2], dim=1) > 0.2), (torch.abs(self.commands[:, 2]) > 0.2))

        contact_mask = is_contact[:, self._feet_contact_ids] # [N_env, N_foot] 
        body_heights = (-self.foot_pos_b[:, :, 2])* contact_mask
        body_heights_sum = torch.sum(body_heights, dim=1)
        contact_count = torch.sum(contact_mask, dim=1) # [N_env] 
        contact_count = torch.clamp(contact_count, min=1.0) # safety
        body_height = body_heights_sum / contact_count
        # body_height = body_heights.min(dim=1).values

        swing_mask = ~contact_mask # [N_env, N_foot] 


        target_min_foot_height = (self.target_min_foot_height * is_commanding).unsqueeze(1)
        min_foot_height_error = torch.square(((target_min_foot_height) - self.peak_foot_heights).clamp(min=0)) # (N,feet)
        # print("error squared", min_foot_height_error[0])
        min_foot_height_error_mapped = torch.sum(torch.exp(-min_foot_height_error / 0.01), dim=-1)
        # min_foot_height_error = torch.sum(torch.square((self.cfg.target_min_foot_height - self.peak_foot_heights).clamp(min=0)), dim=1)
        # min_foot_height_error_mapped = torch.exp(-(min_foot_height_error) / 0.04) 


        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square((self.commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2])), dim=1)
        vel_cmd_magnitude = torch.linalg.norm(self.commands[:, :2], dim=1)
        velocity_scaling_multiple = torch.clamp(1.0 + 0.5 * (vel_cmd_magnitude - 1.0), min=1.0)
        lin_vel_error_mapped_quad = torch.exp(-lin_vel_error / 0.25) * velocity_scaling_multiple
        # yaw rate tracking
        yaw_rate_error = torch.square(self.commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped_quad = torch.exp(-yaw_rate_error / 0.25)
###################################
        stand_up_axis_b = self.target_gravity
        stand_up_axis_b = stand_up_axis_b / torch.linalg.norm(
            stand_up_axis_b, dim=1, keepdim=True
        ).clamp_min(1e-6)

        body_y_axis_b = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        body_z_axis_b = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)

        # 기본 lateral reference
        ref_axis_b = body_y_axis_b

        # target_gravity가 body_y와 너무 평행하면 body_z를 대신 사용
        parallel_to_y = torch.abs(torch.sum(stand_up_axis_b * body_y_axis_b, dim=1)) > 0.95
        ref_axis_b = torch.where(
            parallel_to_y.unsqueeze(1),
            body_z_axis_b,
            body_y_axis_b,
        )

        stand_forward_axis_b = torch.cross(ref_axis_b, stand_up_axis_b, dim=1)
        stand_forward_axis_b = stand_forward_axis_b / torch.linalg.norm(
            stand_forward_axis_b, dim=1, keepdim=True
        ).clamp_min(1e-6)

        stand_lateral_axis_b = torch.cross(stand_up_axis_b, stand_forward_axis_b, dim=1)
        stand_lateral_axis_b = stand_lateral_axis_b / torch.linalg.norm(
            stand_lateral_axis_b, dim=1, keepdim=True
        ).clamp_min(1e-6)

        root_lin_vel_b = self._robot.data.root_lin_vel_b

        stand_forward_vel = torch.sum(root_lin_vel_b * stand_forward_axis_b, dim=1)
        stand_lateral_vel = torch.sum(root_lin_vel_b * stand_lateral_axis_b, dim=1)

        lin_vel_error_standing = torch.sum(
            torch.square(
                torch.stack(
                    [
                        (self.commands[:, 0] / 2.0) - stand_forward_vel,
                        self.commands[:, 1] - stand_lateral_vel,
                    ],
                    dim=1,
                )
            ),
            dim=1,
        )

        vel_cmd_magnitude = torch.linalg.norm(self.commands[:, :2], dim=1)
        velocity_scaling_multiple = torch.clamp(
            1.0 + 0.5 * (vel_cmd_magnitude - 1.0),
            min=1.0,
        )

        lin_vel_error_mapped_standing = torch.exp(
            -lin_vel_error_standing / 0.25
        ) * velocity_scaling_multiple

        stand_turn_rate = torch.sum(self._robot.data.root_ang_vel_b * stand_up_axis_b, dim=1)

        yaw_rate_error_standing = torch.square(self.commands[:, 2] - stand_turn_rate)
        yaw_rate_error_mapped_standing = torch.exp(-yaw_rate_error_standing / 0.25)


        # # linear velocity tracking
        # lin_vel_error_standing = torch.sum(torch.square(torch.stack([(self.commands[:, 0]/2) - self._robot.data.root_lin_vel_b[:, 2], self.commands[:, 1] - self._robot.data.root_lin_vel_b[:, 1]], dim=1)), dim=1)
        # vel_cmd_magnitude = torch.linalg.norm(self.commands[:, :2], dim=1)
        # velocity_scaling_multiple = torch.clamp(1.0 + 0.5 * (vel_cmd_magnitude - 1.0), min=1.0)
        # lin_vel_error_mapped_standing = torch.exp(-lin_vel_error_standing / 0.25) * velocity_scaling_multiple
        # # yaw rate tracking
        # yaw_rate_error_standing = torch.square(self.commands[:, 2] - self._robot.data.root_ang_vel_b[:, 0])
        # yaw_rate_error_mapped_standing = torch.exp(-yaw_rate_error_standing / 0.25)
###################################



        

        lin_vel_error_mapped = torch.where(self.is_standing.bool(), lin_vel_error_mapped_standing, lin_vel_error_mapped_quad)
        yaw_rate_error_mapped = torch.where(self.is_standing.bool(), yaw_rate_error_mapped_standing, yaw_rate_error_mapped_quad)


        pitch_flip_rate = ((self._robot.data.root_ang_vel_b[:, 1]).clamp(min=-0, max=10)) - (torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, [0, 2]]), dim=1))



        body_height = self._robot.data.root_link_pose_w[:,2]

        if self.cfg.terrain_curr_flag:
            body_height = torch.mean(near_base_height[:, :], dim=-1)
        target_base_height = torch.where(self.is_standing.bool(), self.cfg.target_base_height, self.cfg.target_base_height-0.45)
        base_height_error = torch.square(target_base_height - body_height) 
        base_height_error_mapped = 1 - torch.exp(-base_height_error / 0.3)


        quad_base_height_error = torch.square((0.5 - body_height).clamp(min=0))
        quad_base_height_error_mapped = 1 - torch.exp(-quad_base_height_error / 0.1)

        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        
        #foot vel
        feet_vel = torch.sum((torch.linalg.norm(self._robot.data.body_lin_vel_w[:, self._feet_body_ids, :2], dim=2)), dim=-1)
        #foot acceleration
        feet_acc = torch.sum((torch.linalg.norm(self._robot.data.body_lin_acc_w[:, self._feet_body_ids, :2], dim=2)), dim=-1)

        #root center of mass acceleration
        root_com_acc = torch.sum(torch.square(self._robot.data.body_com_acc_w[:,0, :]), dim=1)

        joint_vel = torch.sum(torch.square((self._robot.data.joint_vel)), dim=1)
        # action rate
        diff = torch.square(self._actions[:, :12] - self.last_actions[:, :12])
        diff = diff * (self.last_actions[:, :12] != 0)  # ignore first step
        action_smoothness_1 = torch.sum(diff, dim=1)
        #action_smoothness_2
        diff = torch.square(self._actions[:, :12] - 2 * self.last_actions[:, :12] + self.last_last_actions[:, :12])
        diff = diff * (self.last_actions[:, :12] != 0)  # ignore first step
        diff = diff * (self.last_last_actions[:, :12] != 0)  # ignore second step
        action_smoothness_2 = torch.sum(diff, dim=1)
        action_smoothness = (0.5 * action_smoothness_1) + (10 * action_smoothness_2)
        # feet air time
        
        # last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_contact_ids]
        # air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
        #     torch.norm(self.commands[:, :2], dim=1) > 0.1
        # )
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_contact_ids] 
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * is_commanding.float()

        current_air_time = self._contact_sensor.data.last_air_time[:, self._feet_contact_ids]
        current_contact_time = self._contact_sensor.data.last_contact_time[:, self._feet_contact_ids]

        gait_freq = 2.0
        swing_err  = ((current_air_time - ((1/gait_freq)/2))**2) * is_swing[:, self._feet_contact_ids]
        stance_err = ((current_contact_time - ((1/gait_freq)/2))**2) * is_contact[:, self._feet_contact_ids]
        gait_freq_err_mapped = (torch.sum(torch.exp(-(swing_err + stance_err) / 0.05), dim=-1)/4)

        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_contact_ids]
        last_contact_time = self._contact_sensor.data.last_contact_time[:, self._feet_contact_ids]
        air_time_var = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
            torch.clip(last_contact_time, max=0.5), dim=1
        ) * is_commanding.float()
        # undesired contacts
        air_time = air_time_var
        contacts = torch.sum(is_contact[:, self._undesired_contact_body_ids], dim=1)
        calf_contacts = torch.sum(is_contact[:, self._calf_contact_ids], dim=1)
        # flat orientation


        pg = self._robot.data.projected_gravity_b  # [N, 3]
        # orientation = torch.sum(pg * self.target_gravity, dim=1)

        T = 2.0
        target_pitch_rate = (2.0 * torch.pi / T)

        # ang_vel_w_yaw_aligned = quat_apply_inverse(yaw_quat(self._robot.data.root_quat_w) , self._robot.data.root_ang_vel_w)
        ang_vel_w_yaw_aligned = self._robot.data.root_ang_vel_b
        pitch_rate_error = (target_pitch_rate - ang_vel_w_yaw_aligned[:, 1]).clamp(min=0) ** 2
        pitch_rate_tracking = torch.exp(-pitch_rate_error / 0.5)
        # print(ang_vel_w_yaw_aligned[0, 1])

        # ang_vel_w_yaw_rotated = quat_apply_yaw(self._robot.data.root_quat_w, self._robot.data.root_ang_vel_w)
        # pitch_rate_error = (ang_vel_w_yaw_rotated[:, 1] - target_pitch_rate)**2
        # pitch_rate_tracking = torch.exp(-pitch_rate_error / 0.5)

        # orientation_err = (1-torch.sum(((pg * self.target_gravity)), dim=1))**2
        # orientation = torch.exp(-orientation_err / 1.5)

        # orientation_err = torch.sum((pg * self.target_gravity)**2, dim=1)
        # orientation = torch.exp(-orientation_err / 0.25)

        stand_orientation_error = (1 - (self._robot.data.projected_gravity_b[:, 0])) ** 2
        stand_orientation = torch.exp(-stand_orientation_error / 0.25)


        stand_orientation_error = (1.0 - self._robot.data.projected_gravity_b[:, 0]) ** 2
        stand_orientation_phi = torch.exp(-stand_orientation_error / 0.25)
        stand_orientation_pbrs = (
            0.99 * stand_orientation_phi - self.prev_stand_orientation_phi
        )
        self.prev_stand_orientation_phi[:] = stand_orientation_phi.detach()
        # stand_orientation += (100 * stand_orientation_pbrs)


        quad_orientation_error = (-1 - (self._robot.data.projected_gravity_b[:, 2])) ** 2
        quad_orientation = torch.exp(-quad_orientation_error / 0.25)

        # feet_pose_error = torch.sum(torch.linalg.nom(self.default_feet_pose - self.foot_pos_b[:, :, :2], dim=-1) * is_contact[:, self._feet_contact_ids], dim=1) 
        # feet_pose_error = torch.sum(torch.linalg.norm(self.default_feet_pose - self.foot_pos_b[:, :, :2], dim=-1)* is_contact[:, self._feet_contact_ids], dim=1) * (torch.norm(self.commands[:, :3], dim=1) < 0.2)
        feet_pose_error = torch.sum(torch.norm((self.foot_pos_b[:, :, :2] - self.default_feet_pose), dim=-1), dim=1) + torch.sum(10*torch.abs(self._robot.data.joint_pos[:, :4]), dim=-1)



        # feet_slip = torch.sum(is_contact[:, self._feet_contact_ids] * torch.norm(net_contact_forces_w[:, 0, self._feet_contact_ids, :2], dim=-1), dim=-1) * (torch.norm(self.commands[:, :3], dim=1) > 0.2)

        feet_slip = ((torch.linalg.norm(self._robot.data.body_lin_vel_w[:, self._feet_body_ids, :2], dim=2))*is_contact[:, self._feet_contact_ids]) + torch.abs((self._robot.data.body_ang_vel_w[:, self._feet_body_ids, 2] *is_contact[:, self._feet_contact_ids]))
        # feet_slip = ((torch.linalg.norm(self._robot.data.body_lin_vel_w[:, self._feet_body_ids, :2], dim=2))*is_contact[:, self._feet_contact_ids] )
        feet_slip = torch.sum(feet_slip, dim=-1)
        

        feet_stumble_force = torch.any(torch.norm(net_contact_forces[:,1,self._feet_contact_ids, 0:2],dim=-1) > 5 *torch.abs(net_contact_forces[:, 1,self._feet_contact_ids, 2]), dim=1 )

        stand_still = ~is_commanding * (torch.sum(torch.square(self._robot.data.joint_pos - self._robot.data.default_joint_pos), dim=1) + (0.1 * torch.sum(is_swing[:, self._feet_contact_ids], dim=1)))
        # stand_still = (torch.norm(self.commands[:, :2], dim=1) < 0.2) * torch.sum(torch.norm((self.foot_pos_b[:, :, :2] - self.default_feet_pose), dim=-1), dim=-1)
    
        termination = self.reset_terminated.float()


        q = self._robot.data.joint_pos

        FL_RR_err = torch.cat([
            q[:, [0]] + q[:, [3]],   # hip roll symmetry
            q[:, [4]] - q[:, [7]],   # thigh symmetry
            q[:, [8]] - q[:, [11]],  # calf symmetry
        ], dim=-1)

        FR_RL_err = torch.cat([
            q[:, [1]] + q[:, [2]],
            q[:, [5]] - q[:, [6]],
            q[:, [9]] - q[:, [10]],
        ], dim=-1)

        symmetry = torch.sum(FL_RR_err**2, dim=-1) + torch.sum(FR_RL_err**2, dim=-1)

        c_bool = is_contact[:, self._feet_contact_ids]
        left_equal  = (c_bool[:, 0] == c_bool[:, 3])  # FL == RR
        right_equal = (c_bool[:, 1] == c_bool[:, 2])  # FR== RL
        opposite_between_pairs = (c_bool[:, 0] != c_bool[:, 1])  # (FL,RR) != (FR,RL)
        # contact_mask = c_bool[:, 3] == False
        contact_mask = left_equal & right_equal & opposite_between_pairs

        contact_pattern = contact_mask.float() * is_commanding.float()
        # contact_pattern = contact_mask.float()

        joint_vel_penalty = torch.sum(self.joint_limit_log_barrier(self._robot.data.joint_vel, torch.tensor(-20.0, device=self.device).expand(12), torch.tensor(20.0, device=self.device).expand(12), margin_ratio=0.3), dim=-1)

        margin_ratio = torch.tensor([
            0.15, 0.15, 0.15, 0.15, #HR
            0.05, 0.05, 0.05, 0.05, #HP
            0.05, 0.05, 0.05, 0.05, #KP
        ], device=self.device)
        # print(self._robot.data.projected_gravity_b[0, :])
        joint_penalty = self.joint_limit_log_barrier(self._robot.data.joint_pos, self._robot.data.default_joint_pos_limits[:, :, 0], self._robot.data.default_joint_pos_limits[:, :, 1], margin_ratio=margin_ratio)
        joint_limit_penalty = torch.sum(joint_penalty, dim=1)   # (num_envs,)
        joint_torques_log_barrier = self.joint_limit_log_barrier(self._robot.data.computed_torque, torch.tensor(-100.0, device=self.device).expand(12), torch.tensor(100.0, device=self.device).expand(12), margin_ratio=margin_ratio)
        # print(-self._robot.data.joint_effort_limits[0])
        # print(self._robot.data.computed_torque[0])

        joint_torques_log_barrier_penalty = torch.sum(joint_torques_log_barrier, dim=1)

        


        stand_rewards = {
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            # "feet_slip_l2" : feet_slip * self.cfg.feet_slip_reward_scale *self.step_dt,
            # "joint_vel_penalty": joint_vel_penalty * self.cfg.log_barrier_joint_vel_penalty_scale * self.step_dt,
            # "joint_torques_log_barrier": joint_torques_log_barrier_penalty * self.cfg.log_barrier_joint_torque_penalty_scale * self.step_dt,
            # "action_rate_l2": action_smoothness * self.cfg.action_rate_reward_scale * self.step_dt,
            # "joint_limits": joint_limit_penalty * self.cfg.log_barrier_joint_limit_penalty_scale * self.step_dt,
            "track_base_height_exp" : base_height_error_mapped * self.cfg.base_height_reward_scale * self.step_dt,
            "orientation_l2": stand_orientation * self.cfg.orientation_reward_scale * self.step_dt,
            # "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            # "termination": termination * self.cfg.termination_reward_scale * self.step_dt,
            "stand_orientation_PBRS": stand_orientation_pbrs * self.cfg.standing_orientation_PBRS_scale * self.step_dt,
            # "pitch_flip_rate": pitch_flip_rate * self.cfg.flip_reward_scale * self.step_dt,
        }



        recovery_rewards= {
            "track_base_height_exp" : quad_base_height_error_mapped * self.cfg.base_height_reward_scale * self.step_dt,
            "orientation_l2": quad_orientation * self.cfg.orientation_reward_scale * self.step_dt,
            # "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            # "termination": termination * self.cfg.termination_reward_scale * self.step_dt,
            # "joint_vel_penalty": joint_vel_penalty * self.cfg.log_barrier_joint_vel_penalty_scale * self.step_dt,
            # "action_rate_l2": action_smoothness * self.cfg.action_rate_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            # "joint_limits": joint_limit_penalty * self.cfg.log_barrier_joint_limit_penalty_scale * self.step_dt,
            # "joint_torques_log_barrier": joint_torques_log_barrier_penalty * self.cfg.log_barrier_joint_torque_penalty_scale * self.step_dt,
        }


        quadruped_rewards= {
            "track_lin_vel_xy_exp": lin_vel_error_mapped_quad * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped_quad * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "track_base_height_exp" : quad_base_height_error_mapped * self.cfg.base_height_reward_scale * self.step_dt,
            "orientation_l2": quad_orientation * self.cfg.orientation_reward_scale * self.step_dt,
            "track_foot_height_exp" : min_foot_height_error_mapped * self.cfg.foot_height_reward_scale * self.step_dt,
            "track_gait_frequency": gait_freq_err_mapped * self.cfg.gait_frequency_reward_scale * self.step_dt,
            "feet_slip_l2" : feet_slip * self.cfg.feet_slip_reward_scale *self.step_dt,
            "feet_nominal_pos_l2": feet_pose_error * self.cfg.feet_nominal_pos_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "stand_still" : stand_still * self.cfg.stand_still_reward_scale * self.step_dt,
            "symmetry": symmetry * self.cfg.symetry_reward_scale * self.step_dt,
            "contact_pattern": contact_pattern * self.cfg.contact_pattern_reward_scale * self.step_dt,
            "feet_stumble_force": feet_stumble_force * self.cfg.feet_stumble_force_reward_scale * self.step_dt,
            # "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            # "termination": termination * self.cfg.termination_reward_scale * self.step_dt,
            # "joint_vel_penalty": joint_vel_penalty * self.cfg.log_barrier_joint_vel_penalty_scale * self.step_dt,
            # "action_rate_l2": action_smoothness * self.cfg.action_rate_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            # "joint_limits": joint_limit_penalty * self.cfg.log_barrier_joint_limit_penalty_scale * self.step_dt,
            # "joint_torques_log_barrier": joint_torques_log_barrier_penalty * self.cfg.log_barrier_joint_torque_penalty_scale * self.step_dt,
        }

        biped_rewards = {
            "pitch_flip_rate": pitch_flip_rate * self.cfg.flip_reward_scale * self.step_dt,
            # "track_lin_vel_xy_exp": lin_vel_error_mapped_standing * self.cfg.lin_vel_reward_scale * self.step_dt,
            # "track_ang_vel_z_exp": yaw_rate_error_mapped_standing * self.cfg.yaw_rate_reward_scale * self.step_dt,
            # "feet_slip_l2" : feet_slip * self.cfg.feet_slip_reward_scale *self.step_dt,
            # "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            # "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            # "track_base_height_exp" : base_height_error_mapped * self.cfg.base_height_reward_scale * self.step_dt,
            # "orientation_l2": stand_orientation * self.cfg.orientation_reward_scale * self.step_dt,
        }
        
        safety_rewards = {
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "calf_contacts": calf_contacts * self.cfg.calf_contact_reward_scale * self.step_dt,
            "termination": termination * self.cfg.termination_reward_scale * self.step_dt,
            "joint_vel_penalty": joint_vel_penalty * self.cfg.log_barrier_joint_vel_penalty_scale * self.step_dt,
            "joint_torques_log_barrier": joint_torques_log_barrier_penalty * self.cfg.log_barrier_joint_torque_penalty_scale * self.step_dt,
            "action_rate_l2": action_smoothness * self.cfg.action_rate_reward_scale * self.step_dt,
            "joint_limits": joint_limit_penalty * self.cfg.log_barrier_joint_limit_penalty_scale * self.step_dt,
        }




        # Logging
        for key, value in recovery_rewards.items():
            self._episode_sums[key] += (value)

        for key, value in stand_rewards.items():
            self._episode_sums[key] += (value)

        for key, value in quadruped_rewards.items():
            self._episode_sums[key] += (value)

        for key, value in biped_rewards.items():
            self._episode_sums[key] += (value)

        for key, value in safety_rewards.items():
            self._episode_sums[key] += (value)

        self._episode_sums["alpha"] += (self.alpha.detach().clone()) * self.step_dt
        self._episode_sums["beta"] += (self.beta.detach().clone()) * self.step_dt

        # for key, value in stability_rewards.items():
        #     self._episode_sums[key] += value



        recovery_rewards = torch.sum(torch.stack(list(recovery_rewards.values())), dim=0)
        stand_rewards = torch.sum(torch.stack(list(stand_rewards.values())), dim=0)
        quadruped_rewards = torch.sum(torch.stack(list(quadruped_rewards.values())), dim=0)
        biped_rewards = torch.sum(torch.stack(list(biped_rewards.values())), dim=0)
        safety_rewards = torch.sum(torch.stack(list(safety_rewards.values())), dim=0)


        rewards = {
            "stand_up_critic" : stand_rewards,
            "recovery_critic" : recovery_rewards,
            "quadruped_critic" : quadruped_rewards,
            "biped_critic" : biped_rewards,
            "safety_critic" : safety_rewards
        }
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._terminate_contact_body_ids], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self.peak_foot_heights[env_ids] = 0
        self.last_peak_foot_heights[env_ids] = 0
        self.peak_body_height[env_ids] = 0
        self.peak_foot_vels[env_ids] = 0
        self.last_peak_foot_vels[env_ids] = 0
        self.current_foot_heights[env_ids] = 0
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_last_last_actions[env_ids] = 0.
        self.last_was_contact[env_ids] = 0.
        self.alpha[env_ids] = 0.
        self.beta[env_ids] = 0.
        self.is_standing[env_ids] = 0.
        self.flip_phase[env_ids] = 0.
        # self.prev_stand_orientation_phi[env_ids] = 0.


        if self.cfg.terrain_curr_flag:
            ## terrain curriculm
            distance = torch.norm(self._robot.data.root_pos_w[env_ids, :2] - self._terrain.env_origins[env_ids, :2], dim=1)
            # distance = torch.norm(self._robot.data.root_pos_w[env_ids, :2] - self._terrain.env_origins[env_ids, :2], dim=1)
            # robots that walked far enough progress to harder terrains
            move_up = distance > self._terrain.cfg.terrain_generator.size[0] / 2 # 2
            # robots that walked less than half of their required distance go to simpler terrains
            move_down = distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
            move_down *= ~move_up
            # update terrain levels
            self._terrain.update_env_origins(env_ids, move_up, move_down)



        vel_cmd_magnitude = torch.linalg.norm(self.commands[env_ids, :2], dim=1)  # [N]
        velocity_scaling = torch.clamp(
            1.0 + 0.5 * (vel_cmd_magnitude - 1.0),
            min=1.0,
        )  # [N]

        lin_vel_condition = (
            (self._episode_sums["track_lin_vel_xy_exp"][env_ids]) / self.max_episode_length_s
        ) > (
            self.cfg.speed_curriclum_reward_threshold*2
            # * velocity_scaling
            * self.cfg.lin_vel_reward_scale
        )  # [N] bool

        updated_speed_factor = (
            self.speed_factor[env_ids] + self.cfg.speed_up_factor
        ).clip(self.cfg.init_speed_factor, self.cfg.max_spped)

        self.speed_factor[env_ids] = torch.where(
            lin_vel_condition,
            updated_speed_factor,
            self.speed_factor[env_ids],
        )


        foot_height_condition = (
            self._episode_sums["track_foot_height_exp"][env_ids] / self.max_episode_length_s
        ) > (
            self.cfg.foot_height_curriculum_reward_threshold
            * self.cfg.foot_height_reward_scale
            * 4
        )  # [N] bool

        updated_target_min_foot_height = (
            self.target_min_foot_height[env_ids] + self.cfg.foot_height_up_factor
        ).clip(
            self.cfg.init_target_min_foot_height,
            self.cfg.max_target_min_foot_height,
        )

        self.target_min_foot_height[env_ids] = torch.where(
            foot_height_condition,
            updated_target_min_foot_height,
            self.target_min_foot_height[env_ids],
        )



        self.sample_commands(env_ids)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        if self.common_step_counter > (24*0):
            if self.cfg.events:
                if "reset" in self.event_manager.available_modes:
                    env_step_count = self._sim_step_counter // self.cfg.decimation
                    self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)


        #this reset step is important to prevent reward spike
        stand_orientation_error = (1.0 - self._robot.data.projected_gravity_b[env_ids, 0]) ** 2
        stand_orientation_phi = torch.exp(-stand_orientation_error / 0.25)
        self.prev_stand_orientation_phi[env_ids] = stand_orientation_phi.detach()



        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        if self.cfg.terrain_curr_flag:
            extras["Episode_Termination/mean_terrain_level"] = self._terrain.terrain_levels.float().mean()
        extras["Episode_Termination/mean_speed_factor"] = self.speed_factor.mean().float()
        extras["Episode_Termination/mean_foot_height"] = self.target_min_foot_height.mean().float()
        self.extras["log"].update(extras)

    def sample_commands(self, env_ids):
        # self.commands[env_ids, 0] = torch.zeros_like(self.commands[env_ids, 0]).uniform_(torch.clamp(-speed_factor.mean(), min=-1.0).float(), speed_factor.mean().float())
        # self.commands[env_ids, 0] = torch.zeros_like(self.commands[env_ids, 0]).uniform_(torch.clamp(-speed_factor.mean(), min=-1.0).float(), speed_factor.mean().float())
        self.commands[env_ids, 0] = torch.zeros_like(self.commands[env_ids, 0]).uniform_(-self.speed_factor.mean().float(), self.speed_factor.mean().float())
        self.commands[env_ids, 1] = torch.zeros_like(self.commands[env_ids, 1]).uniform_(torch.clamp(-self.speed_factor.mean(), min=-1.0).float(), torch.clamp(self.speed_factor.mean(), max=1.0).float())
        # self.commands[env_ids, 1] = torch.zeros_like(self.commands[env_ids, 1]).uniform_(-1.0 , 1.0)
        self.commands[env_ids, 2] = torch.zeros_like(self.commands[env_ids, 2]).uniform_((-3.14 * torch.clamp(self.speed_factor.mean(), max=1.0).float()), (3.14 * torch.clamp(self.speed_factor.mean(), max=1.0).float()))
        # self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        # self.commands[env_ids, 2] *= (torch.abs(self.commands[env_ids, 2]) > 0.2)
        self.heading_command[env_ids, 0] = torch.zeros_like(self.heading_command[env_ids, 0]).uniform_(-3.14, 3.14)
        # self.commands[env_ids, :3] *= (torch.norm(self.commands[env_ids, :3], dim=1) > 0.2).unsqueeze(1)

        mask = (torch.rand(len(env_ids), device=self.device) < 0.8) # True/False 50:50
        self.is_standing[env_ids] = mask.clone().detach().float()  # True/False 50:50

        # self.target_gravity = torch.zeros(N, 3, device=self.device)

        local_target = torch.zeros(len(env_ids), 3, device=self.device)
        local_target[mask] = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        local_target[~mask] = torch.tensor([0.0, 0.0, -1.0], device=self.device)

        self.target_gravity[env_ids] = local_target
        
    
    def control_start(self):
        self._robot.actuators["legs"].stiffness = 80 * (self.episode_length_buf > 100).unsqueeze(1)
        self.reward_dict = {key: value * (self.episode_length_buf > 100) for key, value in self.reward_dict.items()}

    def step(self, action: torch.Tensor): 
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)



        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)



        # if self.common_step_counter % 50 == 0:
        #     self.set_sampling_envs(origin_env_id=0, sampling_env_ids=self.sampling_envs_ids)


        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reward_dict = self._get_rewards()
        # self.extras["recoverability"] = self.reset_terminated.detach().clone()
        # self.extras["recoverability"] = torch.logical_or(self.reset_terminated, torch.where(self._robot.data.projected_gravity_b[:, 2] > -0.80, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device)))
        # self.extras["recoverability"] = torch.where(self._robot.data.projected_gravity_b[:, 2] > -0.10, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
        self.extras["recoverability"] = (1-self.alpha.detach().clone()-(self.beta.detach().clone())).clamp(0.0, 1.0)
        self.extras["quad_weight"] = (self.alpha.detach().clone()-self.is_standing.detach().clone()).clamp(0.0, 1.0)
        # self.extras["biped_weight"] = ((1-self.extras["recoverability"]+(self.beta.detach().clone()+self.alpha.detach().clone()))*self.is_standing.detach().clone()).clamp(0.0, 1.0)
        self.extras["biped_weight"] = ((self.alpha + self.beta.detach().clone())*self.is_standing.detach().clone()).clamp(0.0, 1.0)

        self._episode_sums["zrecoverability"] += (self.extras["recoverability"].detach().clone()) * self.step_dt
        self._episode_sums["zquad_weight"] += (self.extras["quad_weight"].detach().clone()) * self.step_dt
        self._episode_sums["zbiped_weight"] += (self.extras["biped_weight"].detach().clone()) * self.step_dt

        if self.common_step_counter > (0):
            self.reset_terminated[:] = False

            # self.cfg.termination_reward_scale  = -5.0
        self.reset_buf[:] = self.reset_terminated | self.reset_time_outs

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        forward = quat_apply(self._robot.data.root_quat_w, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        # 
        # if self.speed_factor.mean() >= (0.7 * self.cfg.max_spped):
        self.commands[:, 2] = (torch.clip(0.75*wrap_to_pi(self.heading_command[:,0] - heading), -1.57 * torch.clamp(self.speed_factor.mean(), max=1.0).float(), 1.57 * torch.clamp(self.speed_factor.mean(), max=1.0).float())) * (torch.abs(self.commands[:, 2]) > 0.1).float()


        # update observations
        self._update_deviations()
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])
            

        # self.control_start()

        # return observations, rewards, resets and extras


        

        return self.obs_buf, TensorDict(self.reward_dict, batch_size=[self.num_envs]), self.reset_terminated, self.reset_time_outs, self.extras