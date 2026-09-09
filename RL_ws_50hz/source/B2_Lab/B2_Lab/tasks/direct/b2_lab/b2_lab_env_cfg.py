import isaaclab.envs.mdp as mdp  # mdp: 강화학습 환경의 행동, 관찰, 보상 함수들
import isaaclab.sim as sim_utils  # sim_utils: 시뮬레이션 관련 유틸리티 (물리, 재질 등)
from isaaclab.assets import ArticulationCfg  # ArticulationCfg: 로봇 관절 시스템 설정
from isaaclab.envs import DirectRLEnvCfg  # DirectRLEnvCfg: 직접 강화학습 환경의 기본 설정 클래스
from isaaclab.managers import EventTermCfg as EventTerm  # EventTerm: 환경 랜덤화 이벤트 설정
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg  # InteractiveSceneCfg: 다중 환경 씬 설정
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns  # ContactSensorCfg: 접촉 센서 설정, RayCasterCfg: 레이 캐스터 센서 설정
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg  # TerrainImporterCfg: 지형 생성 설정
from isaaclab.utils import configclass
from B2_Lab.robots.B2_robot import B2_CFG  # isort: skip
from B2_Lab.terrains import ROUGH_TERRAINS_CFG, CUSTOM_TERRAINS_CFG, CURRICULUM_TERRAINS_CFG  # isort: skip


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.3),
            "num_buckets": 64,
            # 샘플별로 dynamic <= static 을 보장 (물리적 일관성)
            "make_consistent": True,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-5.0, 5.0),  # 더 작은 값으로 변경
            "operation": "add",
        },
    )

    # Actuator gains randomization (privileged obs의 stiffness/damping 항이 이 값을 반영)
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),     # ±10%
            "damping_distribution_params": (0.85, 1.15),     # ±15%
            "operation": "scale",  # 기본값에 곱하기
        },
    )

    # 주기적 외란 (base 속도에 push 주입) — 실기 강건성용, 약하게 설정
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )

    # Joint friction randomization
    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup", 
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 0.05),    # 관절 마찰 추가
            "operation": "add",
        },
    )

    # # Motor strength randomization (추가적인 actuator 변화)
    # randomize_motor_strength = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",  # 에피소드마다 변경
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "stiffness_distribution_params": (0.9, 1.1),    # ±10% 변화 (에피소드마다)
    #         "damping_distribution_params": (0.85, 1.15),    # ±15% 변화
    #         "operation": "scale",
    #     },
    # )


@configclass
class B2LabFlatEnvCfg(DirectRLEnvCfg):

    # env
    episode_length_s = 30.0
    decimation = 4
    action_scale = 0.25
    hip_roll_action_scale_factor = 0.5
    action_space = 12          # B2 : 12-DOF
    observation_space = 49     # = num_proprio (B2)
    state_space = 0
    # action_noise_model = True

    # observation noise: 성분별 물리 단위 std (Gaussian, _init_buffers에서 obs scale을
    # 곱해 scaled space 벡터로 변환). cmd/actions/clock은 내부 생성값이라 노이즈 없음.
    # NoiseModelCfg 대신 직접 적용 — history에 push되기 전에 노이즈를 입혀
    # CENet이 noisy history를 보고 학습하도록 함 (_get_observations 참조).
    add_observation_noise = True

    class noise_std:
        ang_vel = 0.2     # rad/s  (IMU gyro)
        gravity = 0.05    # 단위벡터 성분 (자세 추정 오차)
        joint_pos = 0.01  # rad    (관절 엔코더)
        joint_vel = 1.5   # rad/s  (엔코더 미분)

    # B2 proprio = ang_vel(3)+proj_grav(3)+cmd(3)+joint_pos(12)+joint_vel(12)+actions(12)+clock(4) = 49
    num_proprio = 49
    num_history_len = 10
    # B2 privileged = foot_contact(4)+dyn_fric(4)+act_stiff(12)+act_damp(12)+com_height(1) = 33
    num_privileged_obs = 33
    num_scandots = 100
    num_vel_latent = 3
    num_cenet_latent = 16
    num_priv_encoder_latent = 32
    num_height_encoder_latent = 96
    num_critic_obs = num_proprio + num_vel_latent + num_privileged_obs + num_scandots
    num_actor_obs = num_proprio + num_vel_latent + num_cenet_latent
    # target_height : B2 standing CoM height
    target_height = 0.55

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,   # 200 Hz 물리 (기존 400). decimation=4 → policy 50 Hz
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),  # 중력 명시적 설정
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
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
    #     terrain_generator=CURRICULUM_TERRAINS_CFG,
    #     # use_terrain_origins=True,
    #     max_init_terrain_level=1,   # ✅ 처음엔 row 0~1까지만 스폰 (쉬운 구간)
    #     # env_spacing은 use_terrain_origins=True면 보통 불필요
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     debug_vis=False,
    # )
    # terrain curriculum on/off. plane 지형이면 반드시 False, generator+curriculum 지형이면 True.
    terrain_curriculum = False

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    # events
    events: EventCfg = EventCfg()
    # robot
    robot: ArticulationCfg = B2_CFG.replace(
        spawn=B2_CFG.spawn.replace(activate_contact_sensors=True)
    )
    # B2 USD 는 평평한 계층 → 모든 body 가 /Robot/ 직속. 발도 base_link 밑이 아님.
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True
    )

    # height scanner for terrain perception
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.0, 1.0]),
        mesh_prim_paths=["/World/ground"],
    )

    # foot scanner : 발 순서 [FL, FR, RL, RR] (find_bodies(".*foot.*") 순서와 일치)
    foot_scanner_FL = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/FL_foot",
        ray_alignment="yaw",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.30)),   # 발 위 30cm에서 시작
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.20, 0.20]),
        max_distance=1.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

    foot_scanner_FR = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/FR_foot",
        ray_alignment="yaw",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.30)),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.20, 0.20]),
        max_distance=1.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

    foot_scanner_RL = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/RL_foot",
        ray_alignment="yaw",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.30)),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.20, 0.20]),
        max_distance=1.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

    foot_scanner_RR = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/RR_foot",
        ray_alignment="yaw",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.30)),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.20, 0.20]),
        max_distance=1.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    debug_viz: bool = False
    terrain_curriculum: bool = True
    foot_radius: float = 0.035

    class rewards:
        class scales:
            # 이전 45 kg 로봇 포팅 설정 (비교용으로 보존)
            # tracking_lin_vel = 4.0
            # tracking_foot_pos = 4.0
            # tracking_lin_vel_LPF = 4.0  # 함수/LPF버퍼 삭제됨 — 다시 쓰려면 복구 필요
            # base_height = -5.0
            # tracking_ang_vel = 2.0
            # penalty_ang_vel = -0.5
            # lin_vel_z = -2.0
            # ang_vel_xy = -5.0
            # orientation = -25.0
            # torques = -2.5e-4
            # dof_acc = -2.5e-6
            # dof_vel = -1e-3
            # action_rate = -0.1
            # action_smoothness_1 = -0.1
            # action_smoothness_2 = -0.1
            # joint_deviation_from_default = -0.1
            # contact_vel = -2.5
            # stand_still = -5.0
            # no_slip_vel = -1.0

            # unitree_rl_lab B2 기본 보행 설정
            # tracking_lin_vel = 5.0
            # tracking_ang_vel = 3.0
            tracking_lin_vel = 3.0
            tracking_ang_vel = 5.0
            base_height = -5.0
            lin_vel_z = -6.0
            ang_vel_xy = -10.0  # ang_vel_xy = -30.0
            # roll_orientation = -50.0
            orientation = -50.0
            torques = -1.0e-5
            dof_acc = -2.5e-7
            dof_vel = -1.0e-7
            action_rate = -0.01
            joint_deviation_from_default = -0.7
            # no_slip_vel = -1.5

            # 기본 보행을 먼저 학습하기 위해 중복/커스텀 shaping은 비활성화
            penalty_ang_vel = 0.0
            action_smoothness_1 = 0.0
            action_smoothness_2 = -0.1
            contact_vel = -10.0
            swing_horizontal = -10.0
            stand_still = -5.0

        # 이전 값: tracking_sigma = 0.15
        tracking_sigma = 0.25  # B2 reference std=sqrt(0.25)와 동일한 exp 분모
        sigma_rew_neg = 0.02
        reward_container_name = "B2quadReward"
        kappa_gait_probs = 0.07

    class costs:
        cost_container_name = "B2quadCost"

        class scales:
            c1com_height = 1.0        # 0
            c2dof_pos = 1.0           # 1
            c3dof_vel = 1.0           # 2
            # c4foot_clearance = 1.0
            c5gait_pattern = 1.0      # 3
            c6undesired_contact = 1.0 # 4

    class commands:
        num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        # resampling_time = 20.  # time before command are changed[s]
        heading_command = False  # heading 명령어 사용 여부 (False면 ang_vel_yaw 명령어 사용)
        zero_command_probability = 0.05
        zero_command_threshold = 0.05

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]    # min max [rad/s] # heading 모드용으로 넓게 설정
