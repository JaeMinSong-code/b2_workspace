# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""B2 한 마리만 소환해서 수동 속도 명령으로 걷는 걸 보는 간단 스크립트.

- 로봇 1대만 스폰
- 아래 COMMAND 값(전진/횡/회전)을 코드에서 직접 지정
- logs 안의 '가장 최근 run의 가장 최근 체크포인트'를 자동으로 불러옴
- 비디오/잡다한 argument 전부 제거

실행:  (env_isaaclab_50gpu 활성화 후, scripts/rsl_rl 에서)
    python play_simple.py
"""

# ============================================================
#  여기만 수정하세요 — 수동 속도 명령
COMMAND = [1.0, 0.0, 0.0]   # [전진 vx (m/s), 횡 vy (m/s), 회전 wz (rad/s)]
#  예) 제자리 회전:  [0.0, 0.0, 0.5]     /  후진: [-0.5, 0.0, 0.0]
# ============================================================

import argparse

from isaaclab.app import AppLauncher

# AppLauncher용 인자만 파싱 (--device 등). 나머지는 전부 코드 고정.
parser = argparse.ArgumentParser(description="Simple B2 play with a manual velocity command.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False  # 비디오 안 씀

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import glob
import os
import re
import time

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

TASK = "Unitree-B2-Velocity"
EXPERIMENT = "unitree_b2_velocity"  # logs/rsl_rl/<EXPERIMENT>/


def find_latest_checkpoint() -> str:
    """logs/rsl_rl/<EXPERIMENT>/ 에서 가장 최근 run의 최신 model_*.pt 반환 (실행 위치 기준)."""
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", EXPERIMENT))
    runs = sorted(d for d in glob.glob(os.path.join(log_root, "*")) if os.path.isdir(d))
    if not runs:
        raise FileNotFoundError(f"run 폴더가 없습니다: {log_root}")
    latest_run = runs[-1]  # 날짜 이름이라 알파벳 정렬 = 최신순
    ckpts = glob.glob(os.path.join(latest_run, "model_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"체크포인트가 없습니다: {latest_run}")
    # model_<숫자>.pt 에서 숫자가 가장 큰 것
    return max(ckpts, key=lambda p: int(re.search(r"model_(\d+)", os.path.basename(p)).group(1)))


def main():
    # 환경 설정 (로봇 1대)
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1, entry_point_key="play_env_cfg_entry_point")

    # 명령을 수동값으로 고정: 범위를 (v, v)로 못박고 정지환경 비율 0 → 관측에 항상 COMMAND가 들어감
    vx, vy, wz = COMMAND
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (vx, vx)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (vy, vy)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (wz, wz)

    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")

    resume_path = find_latest_checkpoint()
    print(f"[play_simple] 체크포인트: {resume_path}")
    print(f"[play_simple] 명령: vx={vx}  vy={vy}  wz={wz}")

    # 환경 생성 + rsl-rl 래핑
    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 정책 로드
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt

    # 초기 관측
    obs = env.get_observations()
    if isinstance(obs, tuple):  # rsl-rl 2.3+ 는 (obs, extras) 반환
        obs = obs[0]

    # 실시간 재생 루프
    while simulation_app.is_running():
        t0 = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        sleep_time = dt - (time.time() - t0)
        if sleep_time > 0:
            time.sleep(sleep_time)  # 실시간 속도로 보기

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
