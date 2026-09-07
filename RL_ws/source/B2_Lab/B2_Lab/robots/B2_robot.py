import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# 배포 가능하도록 이 파일 위치 기준 상대 경로로 자산을 참조한다 (하드코딩된 절대경로 금지).
_ROBOTS_DIR = os.path.dirname(os.path.abspath(__file__))
B2_USD_PATH = os.path.join(_ROBOTS_DIR, "assets", "B2", "usd", "b2_flatten.usd")

# Unitree B2 : 12-DOF (다리당 hip/thigh/calf), 관절명은 unitree_ros b2_description 기준.
# 하나의 actuator 그룹으로 12관절 전체를 덮어 env 의 actuator gain 계산 로직과 호환되게 한다.
# effort/velocity limit 은 unitree_rl_lab B2 설정 기준으로 관절별 dict 로 지정.
# armature(로터 반영관성)는 USD(b2_flatten.usd)에 physxJoint:armature=0.1 로 baked 되어 있어
# 여기서 지정하지 않으면 그 USD 값이 사용된다.
B2_ACTUATOR_CFG = DelayedPDActuatorCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    effort_limit={".*_hip_joint": 200.0, ".*_thigh_joint": 200.0, ".*_calf_joint": 320.0},
    velocity_limit={".*_hip_joint": 23.0, ".*_thigh_joint": 23.0, ".*_calf_joint": 14.0},
    stiffness=160.0,
    damping=5.0,
    friction=0.01,
    min_delay=0,
    max_delay=5,
)

B2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=B2_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            fix_root_link=False,
        ),
    ),
    prim_path="/World/envs/env_.*/Robot",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.58),  # B2 standing height
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.7732,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={"legs": B2_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.9,
)
