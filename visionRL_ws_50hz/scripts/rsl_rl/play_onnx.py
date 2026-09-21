import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play with a trained ONNX model.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--plot", action="store_true", default=False, help="Enable real-time plotting of robot data.")
parser.add_argument("--rsl_rl_type", type=str, default="constraints", choices=["original", "constraints"],
                    help="Choose RSL-RL library: 'original' for rsl_rl, 'constraints' for rsl_rl_constraints")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import sys
import time
import torch

# ONNX Runtime import
try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR]: onnxruntime is not installed. Please install it with 'pip install onnxruntime'")
    sys.exit(1)

# 동적으로 RSL-RL 라이브러리 선택
if args_cli.rsl_rl_type == "constraints":
    # rsl_rl_constraints 사용
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../source/rsl_rl_constraints"))
    print("[INFO] Using rsl_rl_constraints library")
else:
    # rsl_rl_2.3.3 사용 (기본값)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../source/rsl_rl_2.3.3"))
    print("[INFO] Using rsl_rl_2.3.3 library")

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import B2_Lab.tasks  # noqa: F401
from B2_Lab.tasks.direct.b2_lab.agents.vecenv_wrapper import RslRlVecEnvWrapper
from B2_Lab.tasks.direct.b2_lab.agents.rsl_rl_ppo_cfg import B2LabFlatPPORunnerCfg
from isaaclab.terrains import TerrainImporterCfg
from B2_Lab.terrains import ROUGH_TERRAINS_CFG, CUSTOM_TERRAINS_CFG, CURRICULUM_TERRAINS_CFG
import isaaclab.terrains as terrain_gen
# Define play terrain configuration
PLAY_TERRAINS_CFG = CUSTOM_TERRAINS_CFG.replace(
    num_rows=2,
    num_cols=2,
    curriculum=False,
)


class ONNXPolicy:
    """ONNX 모델을 사용한 정책 클래스"""
    
    def __init__(self, onnx_path: str, device: str = "cpu"):
        self.device = device
        
        # ONNX 런타임 세션 생성
        providers = ['CPUExecutionProvider']
        if device == "cuda" and ort.get_device() == 'GPU':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # 입력/출력 정보 가져오기
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"[INFO]: ONNX model loaded from: {onnx_path}")
        print(f"[INFO]: Input names: {self.input_names}")
        print(f"[INFO]: Output names: {self.output_names}")
        
        # 입력 형태 정보
        input_shape = self.session.get_inputs()[0].shape
        print(f"[INFO]: Expected input shape: {input_shape}")
    
    def __call__(self, observations):
        """추론 수행"""
        # PyTorch 텐서를 numpy 배열로 변환
        if isinstance(observations, torch.Tensor):
            obs_np = observations.detach().cpu().numpy()
        else:
            obs_np = np.array(observations)
        
        # ONNX 모델 실행
        inputs = {self.input_names[0]: obs_np}
        outputs = self.session.run(self.output_names, inputs)
        
        # 결과를 PyTorch 텐서로 변환
        actions = torch.from_numpy(outputs[0]).to(self.device)
        
        return actions


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: B2LabFlatPPORunnerCfg):
    # ONNX 파일 경로 자동 생성
    log_root_path = os.path.join("logs", "rsl_rl", "1exported")
    log_root_path = os.path.abspath(log_root_path)
    onnx_path = os.path.join(log_root_path, "policy.onnx")
    # ONNX 파일 경로 확인
    if not os.path.exists(onnx_path):
        print(f"[ERROR]: ONNX file not found at: {onnx_path}")
        print("[INFO]: Please run play_b2.py with --convert_onnx first to generate the ONNX model")
        return
    
    # num_envs가 정의되지 않았으면 1로 설정
    if args_cli.num_envs is None:
        args_cli.num_envs = 1
    # task가 정의되지 않았으면 'B2'로 설정
    if args_cli.task is None:
        args_cli.task = "B2"
    
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_generator=PLAY_TERRAINS_CFG,)

    # 환경 생성
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # agent_cfg에서 clip_actions 값 가져오기 (기본값은 1.0)
    clip_actions_value = getattr(agent_cfg, 'clip_actions', 10.0)
    env = RslRlVecEnvWrapper(env, clip_actions=clip_actions_value)

    # ONNX 정책 로드
    device_str = "cuda" if str(env.unwrapped.device).startswith("cuda") else "cpu"
    policy = ONNXPolicy(onnx_path, device=device_str)
    
    dt = env.unwrapped.step_dt

    # 수동 명령 설정
    manual_commands = torch.tensor([-0.0, 0.0, 0.0], device=env.unwrapped.device)
    manual_vxy = torch.tensor([-0.0, 0.0], device=env.unwrapped.device)
    manual_heading = torch.tensor([0.0], device=env.unwrapped.device)
    env.unwrapped._commands[:] = manual_commands.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
    # env.unwrapped._commands[:, 0:2] = manual_vxy.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
    env.unwrapped._heading_cmd[:] = manual_heading.repeat(env.unwrapped.num_envs)

    _, obs_dict = env.get_observations()
    print("[INFO]: Starting ONNX policy inference...")

    try:
        while simulation_app.is_running():
            start_time = time.time()
            # 명령 업데이트
            env.unwrapped._commands[:] = manual_commands.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            # env.unwrapped._commands[:, 0:2] = manual_vxy.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            # env.unwrapped._heading_cmd[:] = manual_heading.repeat(env.unwrapped.num_envs)
            # 관찰 데이터 처리
            prop_obs_history = obs_dict["observations"]["prop_obs_history"]
            actor_obs = prop_obs_history
            with torch.inference_mode():
                # ONNX 정책으로 액션 예측
                actions = policy(actor_obs)
                _, _, _, _, obs_dict = env.step(actions)

            # 실시간 실행 제어
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
   
    except KeyboardInterrupt:
        print("[INFO]: Interrupted by user")
    finally:
        # 정리
        env.close()
        print("[INFO]: Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()
