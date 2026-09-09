import argparse
import sys
from isaaclab.app import AppLauncher
import cli_args

# 명령줄 인자 파싱
parser = argparse.ArgumentParser(description="Play a trained RL agent.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="B2", help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not hasattr(args_cli, "task") or args_cli.task is None:
    args_cli.task = "B2"

# Omniverse 앱 실행
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import time
import torch
import csv
from datetime import datetime

# RSL-RL 라이브러리 로드 (constraints 버전)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../source/rsl_rl_constraints"))

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path
from B2_Lab.tasks.direct.b2_lab.agents.vecenv_wrapper import RslRlVecEnvWrapper
from B2_Lab.tasks.direct.b2_lab.agents.rsl_rl_ppo_cfg import B2LabFlatPPORunnerCfg
from B2_Lab.tasks.direct.b2_lab.b2_lab_env_cfg import B2LabFlatEnvCfg
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from B2_Lab.terrains import ROUGH_TERRAINS_CFG, CUSTOM_TERRAINS_CFG, CURRICULUM_TERRAINS_CFG

def main():
    env_cfg = B2LabFlatEnvCfg()
    agent_cfg = B2LabFlatPPORunnerCfg()
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.experiment_name = "b2_constraints"
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    PLAY_TERRAINS_CFG = CUSTOM_TERRAINS_CFG.replace(num_rows=2, num_cols=2, curriculum=False,)
    env_cfg.terrain = TerrainImporterCfg(prim_path="/World/ground",terrain_generator=PLAY_TERRAINS_CFG,)
    # 환경 생성
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # 모델 로드
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    # 시뮬레이션 설정
    dt = env.unwrapped.step_dt
    # simulation_duration = 40.0  # 시뮬레이션 총 시간 (초)
    simulation_duration = 20.0  # 시뮬레이션 총 시간 (초)
    warmup_time = 5.0  # 초기 대기 시간 (초)
    ramp_duration = 25.0  # 속도 증가 구간 (5~30초 = 25초)
    max_velocity = 1.0  # 최대 x velocity
    _, obs_dict = env.get_observations()
    # CSV 파일 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_dir = os.path.join("outputs", "actions_data")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"robot_data_{timestamp}.csv")
    data_list = []  # 데이터를 저장할 리스트
    step_count = 0
    num_actions = env_cfg.action_space  # 환경 설정에서 액션 차원 수 가져오기
    joint_names = env.unwrapped._robot.data.joint_names
    sim_start_time = time.time()

    # 시뮬레이션 루프
    try:
        while simulation_app.is_running():
            start_time = time.time()
            step_count += 1
            elapsed_time = time.time() - sim_start_time
            if elapsed_time >= simulation_duration:
                print(f"[INFO] Simulation completed: {elapsed_time:.2f} seconds")
                break
            if elapsed_time < warmup_time:
                x_velocity = 0.0
            elif elapsed_time < warmup_time + ramp_duration:
                ramp_progress = (elapsed_time - warmup_time) / ramp_duration
                x_velocity = min(max_velocity, ramp_progress * max_velocity)
            else:
                x_velocity = max_velocity
            
            # manual_commands = torch.tensor([x_velocity, 0.0, 0.0], device=env.unwrapped.device)
            manual_commands = torch.tensor([0.3, 0.0, 0.0], device=env.unwrapped.device)
            manual_vxy = torch.tensor([0.3, 0.0], device=env.unwrapped.device)
            manual_heading = torch.tensor([0.0], device=env.unwrapped.device)
            env.unwrapped._commands[:] = manual_commands.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            # env.unwrapped._commands[:, 0:2] = manual_vxy.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            env.unwrapped._heading_cmd[:] = manual_heading.repeat(env.unwrapped.num_envs)
            # 액션 추론 및 환경 스텝
            with torch.inference_mode():
                # 관측 처리
                prop_obs = obs_dict["observations"]["prop_obs"]
                prop_obs_hist = obs_dict["observations"]["prop_obs_history"]
                v_hat, z_hat = ppo_runner.alg.aux_networks.cenet_infer(prop_obs_hist)
                actor_obs = torch.cat([prop_obs, v_hat, z_hat], dim=-1)
                actions = policy(actor_obs)

                # 데이터 수집
                action_cpu = actions[0].cpu().numpy()
                joint_pos = env.unwrapped._robot.data.joint_pos[0].cpu().numpy()
                joint_vel = env.unwrapped._robot.data.joint_vel[0].cpu().numpy()
                applied_torques = env.unwrapped._robot.data.applied_torque[0].cpu().numpy()
                ref_joint_angles = env.unwrapped._processed_actions[0].cpu().numpy()

                # 추가 데이터 수집
                commands = env.unwrapped._commands[0].cpu().numpy()  # [3] - x, y, yaw
                base_lin_vel = env.unwrapped._robot.data.root_lin_vel_b[0].cpu().numpy()
                base_ang_vel = env.unwrapped._robot.data.root_ang_vel_b[0].cpu().numpy()
                foot_indices = env.unwrapped.foot_indices[0].cpu().numpy()
                clock_inputs = env.unwrapped.clock_inputs[0].cpu().numpy()
                desired_contact_states = env.unwrapped.desired_contact_states[0].cpu().numpy()
                foot_contact_state = env.unwrapped.foot_contact_state[0].cpu().numpy().astype(int)
                # foot_heights = env.unwrapped.foot_pos_b[0, :, 2].cpu().numpy() + env.unwrapped.com_height[0].cpu().numpy() - 0.05  # 월드 좌표계 z
                foot_heights = env.unwrapped.footscanner_height_data[0, :].cpu().numpy()

                # Phases 계산 (cost_foot_clearance에서 사용하는 것과 동일한 공식)
                durations = 0.7  # 또는 self.env에서 가져오기
                swing_start = durations
                swing_phase_mask = env.unwrapped.foot_indices > swing_start
                stance_phase_mask = env.unwrapped.foot_indices <= swing_start
                phases = torch.zeros_like(env.unwrapped.foot_indices)
                swing_progress = (env.unwrapped.foot_indices[swing_phase_mask] - swing_start) / (1.0 - swing_start)  # 0~1
                # 0~0.5 구간: 0에서 1로 증가, 0.5~1 구간: 1에서 0으로 감소
                phases[swing_phase_mask] = torch.where(
                    swing_progress <= 0.5,
                    swing_progress * 2.0,  # 0~0.5 → 0~1
                    2.0 - swing_progress * 2.0  # 0.5~1 → 1~0
                )

                phases[stance_phase_mask] = 0.0
                phases_cpu = phases[0].cpu().numpy() * 0.2  # [4] - FL, FR, HL, HR

                # 데이터 행 생성: [step, timestamp, commands..., actions..., joint_pos..., joint_vel..., torques...]
                data_row = [step_count, time.time()] + \
                           commands.tolist() + \
                           base_lin_vel.tolist() + \
                           base_ang_vel.tolist() + \
                           action_cpu.tolist() + \
                           ref_joint_angles.tolist() + \
                           joint_pos.tolist() + \
                           joint_vel.tolist() + \
                           applied_torques.tolist() + \
                           foot_indices.tolist() + \
                           clock_inputs.tolist() + \
                           desired_contact_states.tolist() + \
                           foot_contact_state.tolist() + \
                           phases_cpu.tolist() + \
                           foot_heights.tolist()
                data_list.append(data_row)
                
                _, _, _, _, obs_dict = env.step(actions)

            # Real-time 실행
            if args_cli.real_time:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        env.close()
        
        # CSV 파일로 저장
        if len(data_list) > 0:
            print(f"[INFO] Saving robot data to: {csv_path}")
            
            # 헤더 생성
            num_joints = len(joint_names)
            header = ['step', 'timestamp']
            # Command 열 이름
            header += ['cmd_x', 'cmd_y', 'cmd_yaw']
            # Base velocity 열 이름
            header += ['base_vel_x', 'base_vel_y', 'base_vel_z']
            header += ['base_ang_vel_x', 'base_ang_vel_y', 'base_ang_vel_z']
            # Action 열 이름
            header += [f'action{i}' for i in range(num_actions)]
            # Reference joint angle 열 이름
            header += [f'ref_pos{i}' for i in range(num_joints)]
            # Joint position 열 이름
            header += [f'pos{i}' for i in range(num_joints)]
            # Joint velocity 열 이름
            header += [f'vel{i}' for i in range(num_joints)]
            # Torque 열 이름
            header += [f'torque{i}' for i in range(num_joints)]
            # Foot indices 열 이름 (FL, FR, HL, HR)
            header += ['foot_idx_FL', 'foot_idx_FR', 'foot_idx_HL', 'foot_idx_HR']
            # Clock inputs 열 이름 (FL, FR, HL, HR)
            header += ['clock_FL', 'clock_FR', 'clock_HL', 'clock_HR']
            # Desired contact states 열 이름 (FL, FR, HL, HR)
            header += ['desired_contact_FL', 'desired_contact_FR', 'desired_contact_HL', 'desired_contact_HR']
            # Foot contact state 열 이름 (FL, FR, HL, HR)
            header += ['contact_FL', 'contact_FR', 'contact_HL', 'contact_HR']
            # Phases 열 이름 (FL, FR, HL, HR)
            header += ['phase_FL', 'phase_FR', 'phase_HL', 'phase_HR']
            # Foot heights 열 이름 (FL, FR, HL, HR)
            header += ['foot_height_FL', 'foot_height_FR', 'foot_height_HL', 'foot_height_HR']

            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                writer.writerows(data_list)
            
            print(f"[INFO] Saved {len(data_list)} data samples to {csv_path}")
        else:
            print("[WARNING] No data to save")


if __name__ == "__main__":
    main()
    simulation_app.close()



