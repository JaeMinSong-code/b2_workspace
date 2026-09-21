# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseModel, NoiseCfg, GaussianNoiseCfg, UniformNoiseCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.rbq10 import RBQ10_CFG, RBQ10_TWO_STAND_CFG, RBQ10_TWO_STAND_REVERSE_CFG  # isort: skip
from isaaclab.terrains.config.rough import KALE_TERRAINS_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )

    resample_command = EventTerm(
        func=mdp.resample_command,
        mode="interval",
        interval_range_s=(2.0, 18.0),
    )

    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     is_global_time=False,
    #     interval_range_s=(2.0, 18.0),
    #     params={"velocity_range": {"x": (-2.5, 2.5), "y": (-2.5, 2.5), "z": (-0.1, 0.1) }}#,"pitch": (-3.14, 3.14)
    # )


    # reset_base = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         #
    #         "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (-0, 0.9), "yaw": (-3.14, 3.14), "roll": (-3.0, 3.0), "pitch": (-3.0, 3.0)},
    #         "velocity_range": {
    #             "x": (-0.0, 0.0),
    #             "y": (-0.0, 0.0),
    #             "z": (-0.0, 0.0),
    #             "roll": (-0.0, 0.0),
    #             "pitch": (-0.0, 0.0),
    #             "yaw": (-0.0, 0.0),
    #         },
    #     },
    # )



@configclass
class RBQEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 4
    action_scale = 0.25
    action_space = 12
    observation_space = 45
    state_space = 0

    terrain_curr_flag = False

    #gait parameters
    init_speed_factor = 0.5 # m/s
    speed_up_factor = 0.3 # m/s
    max_spped = 3.0 # m/s
    speed_curriclum_reward_threshold = 0.8 # ratio of tracking reward

    target_base_height = 1.1

    init_target_min_foot_height = 0.03
    foot_height_up_factor = 0.01
    foot_height_curriculum_reward_threshold = 0.8 # ratio of tracking reward
    max_target_min_foot_height = 0.12

    ## !!caution!! not to confused between _robot body order and contact sensor body order
    #in RegEx form
    termination_contact_body_ids_list = ["base", ".*hip"] #["base",".*hip"]
    undesired_contact_body_ids_list = [".*thigh"] #["base",".*hip"]
    calf_contact_body_ids_list = [".*calf"]

    # reward scales
    z_vel_reward_scale = -4.0
    ang_vel_reward_scale = -0.5
    joint_torque_reward_scale = -1.0e-6
    joint_accel_reward_scale = -2.5e-7 #-2.5e-6
    log_barrier_joint_vel_penalty_scale = -1.0
    log_barrier_joint_limit_penalty_scale = -1.0
    log_barrier_joint_torque_penalty_scale = -1.0
    action_rate_reward_scale = -0.01#-0.01
    
    stand_still_reward_scale = -5.0
    feet_air_time_reward_scale = -5.0

    feet_nominal_pos_reward_scale = -0.1
    feet_slip_reward_scale = -0.5
    symetry_reward_scale = -0.5 #-2.5

    lin_vel_reward_scale = 8.0
    yaw_rate_reward_scale = 4.0
    flip_reward_scale = 10.0
    no_feet_contact_reward_scale = -1.0
    orientation_reward_scale = 8.0
    standing_orientation_PBRS_scale = 100.0 #1000
    
    contact_pattern_reward_scale= 2.0
    base_height_reward_scale = -8.0
    foot_height_reward_scale = 1.0
    gait_frequency_reward_scale = 1.0

    feet_stumble_force_reward_scale = -0.0# -2.5
    
    termination_reward_scale = -10.0#-500.0  
    undesired_contact_reward_scale = -5.0
    calf_contact_reward_scale = -3.0

    sim_freq = 200
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / sim_freq,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx = sim_utils.PhysxCfg(
            # gpu_collision_stack_size=2**26, #for rough terrain
            gpu_max_rigid_patch_count=2**19, #for rough terrain
            gpu_max_num_partitions=16,
            gpu_total_aggregate_pairs_capacity = 2**23,
            gpu_collision_stack_size = 2**27,
            gpu_heap_capacity = 2**27,
        )
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     terrain_generator=KALE_TERRAINS_CFG,
    #     max_init_terrain_level=5,
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    #     visual_material=sim_utils.MdlFileCfg(
    #         mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
    #         project_uvw=True,
    #     ),
    #     debug_vis=False,
    # )
    # terrain_curr_flag = True
    # feet_slip_reward_scale = -0.01


    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True, filter_collisions=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = RBQ10_TWO_STAND_REVERSE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base/.*", history_length=3, update_period=1/sim_freq, track_air_time=True
    )

    projected_gravity_noise_cfg = UniformNoiseCfg(operation="add",n_min=-0.1, n_max=0.1)
    ang_vel_noise_cfg = UniformNoiseCfg(operation="add",n_min=-0.1, n_max=0.1)
    joint_pos_noise_cfg = UniformNoiseCfg(operation="add",n_min=-0.05, n_max=0.05)
    joint_vel_noise_cfg = UniformNoiseCfg(operation="add",n_min=-1.5, n_max=1.5)
    height_map_noise_cfg = UniformNoiseCfg(operation="add",n_min=-0.01, n_max=0.01)
    command_noise_cfg = UniformNoiseCfg(operation="add",n_min=-0.01, n_max=0.01)

    height_scanner_sparse_cfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.5, size=[6.0, 3.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=(1/sim_freq)*decimation
    )

    height_scanner_dense_cfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=(1/sim_freq)*decimation
    )


    foot_scanner_RR = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base/RR_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.01, 0.01]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=(1/sim_freq)*decimation
    )

    foot_scanner_RL = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base/RL_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.01, 0.01]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=(1/sim_freq)*decimation
    )
    foot_scanner_FR = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base/FR_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.01, 0.01]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=(1/sim_freq)*decimation
    )

    foot_scanner_FL = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base/FL_foot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.01, 0.01]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=(1/sim_freq)*decimation
    )