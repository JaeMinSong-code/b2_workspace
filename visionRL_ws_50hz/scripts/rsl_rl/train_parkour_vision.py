"""Extreme Parkour — Phase 2 (vision student distillation).

Loads the Phase-1 teacher (ActorCriticParkour trained with `train_b2.py --parkour`)
and distills it into a depth-based student: a CNN+GRU depth backbone predicts the
teacher's scandots latent (+ yaw) from the forward depth image, and the student
actor is trained to match the teacher's actions.

Losses (per Cheng et al., "Extreme Parkour with Legged Robots"):
  * action distillation : MSE(student_action, teacher_action)
  * scandots-latent      : MSE(depth_scan_latent, teacher_scan_latent)
  * yaw prediction       : MSE(depth_yaw, [delta_yaw, delta_next_yaw])

Run (cameras must be enabled for the depth sensor):
  python train_parkour_vision.py --num_envs 256 --enable_cameras \
      --teacher_ckpt /path/to/phase1/model_XXXX.pt

NOTE: This trains from `num_envs` parallel envs. Depth rendering is expensive, so
use far fewer envs than Phase 1 (e.g. 128-256). Not yet validated end-to-end in
sim — see EXTREME_PARKOUR_PLAN.md for the checklist.
"""

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Extreme Parkour vision student distillation.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments (keep small; depth is costly).")
parser.add_argument("--task", type=str, default="B2-Parkour", help="Task id.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--teacher_ckpt", type=str, required=True, help="Phase-1 teacher checkpoint (.pt).")
parser.add_argument("--max_iterations", type=int, default=5000)
parser.add_argument("--num_steps_per_env", type=int, default=24)
parser.add_argument("--num_pretrain_iter", type=int, default=0,
                    help="Iterations to roll out with TEACHER actions before switching to student.")
parser.add_argument("--lr", type=float, default=1.0e-3)
parser.add_argument("--hist_encoding", action="store_true",
                    help="Use the history (RMA) encoder for the priv latent in the student "
                         "(deployable path; requires a phase-1 RMA-trained history encoder).")
parser.add_argument("--terrain", action="store_true",
                    help="Use the obstacle-terrain env (match a terrain-trained teacher). "
                         "Recommended when the teacher was trained with --terrain.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Depth camera needs the RTX sensor pipeline.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import torch
import torch.nn.functional as F
import gymnasium as gym

# Extreme Parkour uses the rsl_rl_2.3.3 (original) library.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../source/rsl_rl_2.3.3"))
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import RecurrentDepthBackbone

from B2_Lab.tasks.direct.b2_lab.agents.rsl_rl_ppo_cfg import B2LabParkourPPORunnerCfg
from B2_Lab.tasks.direct.b2_lab.b2_lab_parkour_env_cfg import B2LabParkourEnvCfg, B2LabParkourTerrainEnvCfg
from B2_Lab.tasks.direct.b2_lab.agents.vecenv_wrapper import RslRlVecEnvWrapper
import B2_Lab.tasks  # noqa: F401

# Proprio index of the (delta_yaw, delta_next_yaw) slots — see env _get_observations.
YAW_SLICE = slice(7, 9)


def main():
    device = "cuda:0"
    agent_cfg = B2LabParkourPPORunnerCfg()
    agent_cfg.seed = args_cli.seed

    env_cfg = B2LabParkourTerrainEnvCfg() if args_cli.terrain else B2LabParkourEnvCfg()
    env_cfg.enable_depth = True  # phase-2 needs the depth camera
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- teacher (frozen) -------------------------------------------------
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args_cli.teacher_ckpt)
    teacher = runner.alg.policy
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # --- student depth backbone ------------------------------------------
    depth = RecurrentDepthBackbone(
        num_prop=env_cfg.num_prop_pk,
        scandots_latent_dim=agent_cfg.policy.scan_encoder_dims[-1],
        depth_height=env_cfg.depth_height,
        depth_width=env_cfg.depth_width,
    ).to(device)
    optimizer = torch.optim.Adam(depth.parameters(), lr=args_cli.lr)

    log_dir = os.path.join("logs", "rsl_rl", "b2_parkour_vision", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] logging to {log_dir}")

    _, extras = env.get_observations()
    obs_dict = extras["observations"]

    for it in range(args_cli.max_iterations):
        agg = {"action": 0.0, "scan": 0.0, "yaw": 0.0}
        for _ in range(args_cli.num_steps_per_env):
            policy_obs = obs_dict["policy"].to(device)
            proprio = obs_dict["prop_obs_parkour"].to(device)
            depth_img = obs_dict["depth"].to(device)

            with torch.no_grad():
                teacher_scan_latent = teacher.infer_scandots_latent(policy_obs)
                teacher_action = teacher.act_inference(policy_obs)

            # Student depth encoder -> predicted scan latent + yaw.
            depth_out = depth(depth_img, proprio)
            scan_latent_pred = depth_out[:, :-2]
            yaw_pred = depth_out[:, -2:]

            # Student actor: teacher backbone fed with the predicted scan latent
            # and proprio whose yaw slots are replaced by the predicted yaw.
            obs_student = policy_obs.clone()
            obs_student[:, YAW_SLICE] = yaw_pred.detach()
            student_action = teacher.act_inference(
                obs_student, hist_encoding=args_cli.hist_encoding, scan_latent=scan_latent_pred
            )

            # Losses.
            yaw_target = policy_obs[:, YAW_SLICE]
            loss_action = F.mse_loss(student_action, teacher_action)
            loss_scan = F.mse_loss(scan_latent_pred, teacher_scan_latent)
            loss_yaw = F.mse_loss(yaw_pred, yaw_target)
            loss = loss_action + loss_scan + loss_yaw

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(depth.parameters(), 1.0)
            optimizer.step()

            # Which actions to actually execute in the env.
            if it < args_cli.num_pretrain_iter:
                step_action = teacher_action.detach()
            else:
                step_action = student_action.detach()

            _, _, _, dones, extras = env.step(step_action)
            obs_dict = extras["observations"]

            # Truncated BPTT: detach GRU state, and clear it for envs that reset.
            depth.detach_hidden_states()
            depth.reset(dones)

            agg["action"] += loss_action.item()
            agg["scan"] += loss_scan.item()
            agg["yaw"] += loss_yaw.item()

        n = args_cli.num_steps_per_env
        print(f"[it {it:04d}] action={agg['action']/n:.4f} scan={agg['scan']/n:.4f} yaw={agg['yaw']/n:.4f}")
        if (it + 1) % 100 == 0:
            path = os.path.join(log_dir, f"depth_{it+1}.pt")
            torch.save({"depth_encoder": depth.state_dict()}, path)
            print(f"[INFO] saved {path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
