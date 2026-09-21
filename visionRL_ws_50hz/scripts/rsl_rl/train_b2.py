import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train B2 Lab with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="B2", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=50000, help="RL Policy training iterations.")
parser.add_argument("--rsl_rl_type", type=str, default="constraints", choices=["original", "constraints"],
                    help="Choose RSL-RL library: 'original' for rsl_rl, 'constraints' for rsl_rl_constraints")
parser.add_argument("--parkour", action="store_true",
                    help="Train the Extreme Parkour teacher (B2-Parkour task + ActorCriticParkour). "
                         "Forces --rsl_rl_type original.")
parser.add_argument("--terrain", action="store_true",
                    help="With --parkour: use the obstacle-terrain variant (stairs/boxes/slopes "
                         "+ curriculum) instead of a flat plane.")
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not hasattr(args_cli, "task") or args_cli.task is None:
    args_cli.task = "B2"
# Parkour requires the rsl_rl_2.3.3 (original) library and its own task id.
if args_cli.parkour:
    args_cli.rsl_rl_type = "original"
    if args_cli.task == "B2":
        args_cli.task = "B2-Parkour"
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import sys

# 동적으로 RSL-RL 라이브러리 선택
if args_cli.rsl_rl_type == "constraints":
    # 원본 rsl_rl 사용
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../source/rsl_rl_constraints"))
    print("[INFO] Using rsl_rl_constraints rsl_rl library")
else:
    # rsl_rl_constraints 사용 (기본값)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../source/rsl_rl_2.3.3"))
    print("[INFO] Using rsl_rl_2.3.3 library")

from rsl_rl.runners import OnPolicyRunner
from B2_Lab.tasks.direct.b2_lab.agents.rsl_rl_ppo_cfg import B2LabFlatPPORunnerCfg, B2LabParkourPPORunnerCfg
from B2_Lab.tasks.direct.b2_lab.b2_lab_env_cfg import B2LabFlatEnvCfg
from B2_Lab.tasks.direct.b2_lab.b2_lab_parkour_env_cfg import B2LabParkourEnvCfg, B2LabParkourTerrainEnvCfg
from B2_Lab.tasks.direct.b2_lab.agents.vecenv_wrapper import RslRlVecEnvWrapper
import B2_Lab.tasks  # noqa: F401


def main():
    # task가 정의되지 않았으면 'B2'로 설정
    if not hasattr(args_cli, "task") or args_cli.task is None:
        args_cli.task = "B2"
    # create configuration
    if args_cli.parkour:
        agent_cfg = B2LabParkourPPORunnerCfg()
        env_cfg = B2LabParkourTerrainEnvCfg() if args_cli.terrain else B2LabParkourEnvCfg()
        env_cfg.enable_depth = False  # phase-1 teacher: no depth camera (faster, no --enable_cameras needed)
    else:
        agent_cfg = B2LabFlatPPORunnerCfg()
        env_cfg = B2LabFlatEnvCfg()
    agent_cfg.seed = args_cli.seed
    agent_cfg.max_iterations = args_cli.max_iterations

    # create environment configuration
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 로그 디렉토리에 사용된 라이브러리 정보 포함
    lib_type = args_cli.rsl_rl_type
    log_dir = os.path.join("logs", "rsl_rl", f"b2_{lib_type}", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device="cuda:0")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
