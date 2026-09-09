import argparse
from glob import glob
import sys
from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--plot", action="store_true", default=False, help="Enable real-time plotting of robot data.")
parser.add_argument("--convert_onnx", action="store_true", default=False, help="Convert loaded model to ONNX format after loading.")
parser.add_argument("--rsl_rl_type", type=str, default="constraints", choices=["original", "constraints"],
                    help="Choose RSL-RL library: 'original' for rsl_rl, 'constraints' for rsl_rl_constraints")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not hasattr(args_cli, "task") or args_cli.task is None:
    args_cli.task = "B2"
if args_cli.num_envs is None:
        args_cli.num_envs = 1
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import sys
import time
import torch

# 학습에 사용한 RSL-RL 구현을 checkpoint 로드 전에 명시적으로 선택한다.
if args_cli.rsl_rl_type == "constraints":
    rsl_rl_source_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../source/rsl_rl_constraints")
    )
else:
    rsl_rl_source_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../source/rsl_rl_2.3.3")
    )
sys.path.insert(0, rsl_rl_source_dir)

import rsl_rl
from rsl_rl.runners import OnPolicyRunner

loaded_rsl_rl_path = os.path.realpath(rsl_rl.__file__)
expected_rsl_rl_path = os.path.realpath(os.path.join(rsl_rl_source_dir, "rsl_rl"))
if os.path.commonpath([loaded_rsl_rl_path, expected_rsl_rl_path]) != expected_rsl_rl_path:
    raise RuntimeError(
        f"Wrong rsl_rl package loaded: {loaded_rsl_rl_path}. "
        f"Expected a package under: {expected_rsl_rl_path}"
    )
print(f"[INFO] Using {args_cli.rsl_rl_type} rsl_rl library: {loaded_rsl_rl_path}")

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import B2_Lab.tasks  # noqa: F401
from B2_Lab.tasks.direct.b2_lab.agents.vecenv_wrapper import RslRlVecEnvWrapper
from B2_Lab.tasks.direct.b2_lab.agents.rsl_rl_ppo_cfg import B2LabFlatPPORunnerCfg
from exporter import export_policy_as_onnx
from play_logger import PlayLogger
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from B2_Lab.terrains import (
    ROUGH_TERRAINS_CFG,
    CUSTOM_TERRAINS_CFG,
    CURRICULUM_TERRAINS_CFG,
    PLAY_TERRAINS_CFG,
)
import isaaclab.terrains as terrain_gen


def _find_logs_rsl_rl_dir(checkpoint_path: str) -> str:
    p = os.path.abspath(checkpoint_path)
    cur = os.path.dirname(p)  # run dir부터 위로 탐색
    while True:
        # cur == .../logs/rsl_rl 인 순간을 잡는다: basename(cur)=rsl_rl and parent=logs
        if os.path.basename(cur) == "rsl_rl" and os.path.basename(os.path.dirname(cur)) == "logs":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise RuntimeError(f"Could not find '/logs/rsl_rl' in checkpoint_path: {checkpoint_path}")


def export_policy_to_onnx(
    ppo_runner,
    env,
    checkpoint_path: str,
    filename="policy.onnx",
    opset_version=11,
):
    print("[INFO]: Converting policy to ONNX format...")

    try:
        checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # run 폴더 이름 (예: 2026-01-04_12-26-46)
        run_dir = os.path.dirname(checkpoint_path)
        run_name = os.path.basename(run_dir)

        # export base = .../logs/rsl_rl
        logs_rsl_rl_dir = _find_logs_rsl_rl_dir(checkpoint_path)

        # 최종 경로: .../logs/rsl_rl/1exported/<run_name>/policy.onnx
        export_dir = os.path.join(logs_rsl_rl_dir, "1exported", run_name)
        os.makedirs(export_dir, exist_ok=True)
        onnx_path = os.path.join(export_dir, filename)

        P = int(env.cfg.num_proprio)
        H = int(env.cfg.num_history_len)
        input_size = P * H

        class PolicyWrapper(torch.nn.Module):
            def __init__(self, actor, aux_networks, num_proprio: int, num_history_len: int):
                super().__init__()
                self.actor = actor
                self.aux_networks = aux_networks
                self.P = int(num_proprio)
                self.H = int(num_history_len)

            def forward(self, prop_obs_history: torch.Tensor):
                # prop_obs_history: [B, H*P] (flatten), newest = last block
                prop_obs = prop_obs_history[:, (self.H - 1) * self.P : self.H * self.P]
                v_hat, z_hat = self.aux_networks.cenet_infer(prop_obs_history)
                actor_obs = torch.cat([prop_obs, v_hat, z_hat], dim=-1)
                return self.actor(actor_obs)

        wrapped_policy = PolicyWrapper(
            ppo_runner.alg.actor,
            ppo_runner.alg.aux_networks,
            num_proprio=P,
            num_history_len=H,
        ).eval()

        dummy_input = torch.randn(1, input_size, device=env.device)

        torch.onnx.export(
            wrapped_policy,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["prop_obs_history"],
            output_names=["actions"],
            dynamic_axes={
                "prop_obs_history": {0: "batch_size"},
                "actions": {0: "batch_size"},
            },
            verbose=True,
        )

        print(f"[INFO]: Using checkpoint: {checkpoint_path}")
        print(f"[INFO]: Policy successfully exported to ONNX at: {onnx_path}")
        return onnx_path

    except Exception as e:
        print(f"[ERROR]: Failed to export policy to ONNX: {e}")
        import traceback
        traceback.print_exc()
        return None


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: B2LabFlatPPORunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.experiment_name = "b2_constraints"
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Override terrain configuration for play
    # 기본: 난이도 중간대로 좁힌 play 전용 소형 지형. 학습 terrain의 prim_path/physics_material 재사용.
    env_cfg.terrain.terrain_generator = PLAY_TERRAINS_CFG
    env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.terrain_curriculum = False  # 평가 중 terrain level 진급 끔

    # 실제 학습에 사용한 지형(전체 난이도 range)으로 play하려면 위 3줄 대신 아래 사용:
    # env_cfg.terrain.terrain_generator = CURRICULUM_TERRAINS_CFG
    # env_cfg.terrain.max_init_terrain_level = 9   # 보고 싶은 난이도 레벨(0~9)
    # env_cfg.terrain_curriculum = False           # 평가 중 진급은 끔

    # env_cfg.debug_viz = True
    # env_cfg.foot_scanner_HR.debug_vis = True
    # env_cfg.foot_scanner_HL.debug_vis = True
    # env_cfg.foot_scanner_FR.debug_vis = True
    # env_cfg.foot_scanner_FL.debug_vis = True
    # env_cfg.height_scanner.debug_vis = True

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    dt = env.unwrapped.step_dt

    # ONNX 변환 (--convert_onnx 옵션이 있을 때만)
    if args_cli.convert_onnx:
        export_policy_to_onnx(
            ppo_runner=ppo_runner,
            env=env,
            checkpoint_path=resume_path,
        )

    manual_commands = torch.tensor([1.0, 0.0, 0.0], device=env.unwrapped.device)
    manual_vxy = torch.tensor([0.5, 0.0], device=env.unwrapped.device)
    manual_heading = torch.tensor([0.0], device=env.unwrapped.device)
    env.unwrapped._commands[:] = manual_commands.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
    # env.unwrapped._commands[:, 0:2] = manual_vxy.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
    env.unwrapped._heading_cmd[:] = manual_heading.repeat(env.unwrapped.num_envs)

    step_count = 0  # 스텝 카운터 추가````

    # 재생 데이터 로깅 (종료 시 checkpoint run 폴더에 PNG 저장)
    data_logger = PlayLogger(env.unwrapped)

    _, obs_dict = env.get_observations()

    try:
        while simulation_app.is_running():
            start_time = time.time()
            step_count += 1  # 스텝 카운트 증가
            env.unwrapped._commands[:] = manual_commands.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            # env.unwrapped._commands[:, 0:2] = manual_vxy.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            # env.unwrapped._heading_cmd[:] = manual_heading.repeat(env.unwrapped.num_envs)

            with torch.inference_mode():
                prop_obs = obs_dict["observations"]["prop_obs"]
                prop_obs_hist = obs_dict["observations"]["prop_obs_history"]
                # priv_latent = ppo_runner.alg.aux_networks.infer_priv_latent(prop_obs_hist)
                # velocity_latent = ppo_runner.alg.aux_networks.infer_velocity(prop_obs_hist)
                v_hat, z_hat = ppo_runner.alg.aux_networks.cenet_infer(prop_obs_hist)
                actor_obs = torch.cat([prop_obs, v_hat, z_hat], dim=-1)
                actions = policy(actor_obs)
                _, _, _, _, obs_dict = env.step(actions)

            data_logger.log_step(step_count * dt)

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        plot_path = data_logger.save(
            out_dir=os.path.join(os.path.dirname(resume_path), "play_plots"),
            target_height=env.unwrapped.cfg.target_height,
        )
        if plot_path is not None:
            print(f"[play_b2] 플롯 저장: {plot_path}")
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
