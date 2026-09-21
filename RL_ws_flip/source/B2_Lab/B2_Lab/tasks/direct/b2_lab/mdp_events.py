"""B2 전용 커스텀 MDP 이벤트 항.

목적: 관절 plant 점성마찰(Fv, N·m·s/rad) 주입.

배경:
    Isaac Lab 기본 ``randomize_joint_parameters`` 는 joint friction(계수)/armature/limit 만
    랜덤화하고 damping 은 다루지 않는다. 또 ``randomize_actuator_gains`` 의 damping 은
    explicit actuator(여기선 DelayedPD)에서 PD 컨트롤러 Kd(파이썬 계산 토크)에만 반영된다.

    DelayedPD(explicit) actuator 는 초기화 시 sim 드라이브 stiffness/damping 이 0 으로 세팅되고
    (articulation._process_actuators_cfg), 제어토크는 effort 로 별도 인가된다. 따라서
    ``write_joint_damping_to_sim`` 으로 sim 관절 damping 을 넣으면 PhysX 가 이를 목표속도 0 의
    드라이브 damping, 즉 ``-Fv*dq`` 순수 점성마찰로 적용한다. 이 값은 PD Kd 와 완전히 독립이다.

    ⚠️ PhysX 백엔드 전용 동작(effort 인가 + 드라이브 damping 의 가법 합산 가정)이므로,
      Newton/다른 백엔드로 옮기면 재검증 필요.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv


def randomize_joint_damping(
    env: "DirectRLEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    damping_range: tuple[float, float],
):
    """지정 관절의 sim 관절 damping(=plant 점성마찰 Fv, N·m·s/rad)을 ``damping_range`` 에서
    균등 샘플해 물리엔진에 직접 설정한다.

    PD 컨트롤러 Kd(actuator.damping)와는 무관하게 드라이브 damping 을 절대값(abs)으로 덮어쓴다.
    ``startup`` 모드로 등록하면 시작 시 1회 샘플되어 학습 내내 유지된다.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 환경 인덱스 해석
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # 관절 인덱스 해석 (전체면 slice, 아니면 텐서)
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
        num_joints = asset.num_joints
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.long, device=asset.device)
        num_joints = len(asset_cfg.joint_ids)

    lo, hi = damping_range
    damping = math_utils.sample_uniform(lo, hi, (len(env_ids), num_joints), device=asset.device)

    # write_joint_damping_to_sim 은 actuator 내부 버퍼(PD Kd)는 건드리지 않고 sim 만 갱신한다.
    asset.write_joint_damping_to_sim(damping, joint_ids=joint_ids, env_ids=env_ids)
