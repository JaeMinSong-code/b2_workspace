"""Export a B2 CENet checkpoint as a self-contained deployment bundle."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import pickle
import re
import sys
import tempfile
from typing import Any

import torch

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


_SCRIPT_PATH = Path(__file__).resolve()
_SCRIPT_DIR = _SCRIPT_PATH.parent
_RL_ROOT = _SCRIPT_PATH.parents[2]
_DEFAULT_EXPERIMENT = "b2_constraints"
_DEFAULT_LOG_ROOT = _SCRIPT_DIR / "logs" / "rsl_rl"
_CHECKPOINT_PATTERN = re.compile(r"model_(\d+)\.pt$")


parser = argparse.ArgumentParser(description="Export the B2 policy for pKIRO deployment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments used to restore the policy.")
parser.add_argument("--task", type=str, default="B2", help="Isaac Lab task name.")
parser.add_argument("--output-dir", type=str, default=None, help="Deployment bundle directory.")
parser.add_argument("--opset", type=int, default=11, help="ONNX opset version.")
parser.add_argument(
    "--rsl_rl_type",
    type=str,
    default="constraints",
    choices=["original", "constraints"],
    help="RSL-RL implementation used by the checkpoint.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# Deployment never renders a scene. Keep Isaac Sim headless even when the CLI
# flag is omitted so the default export path does not open a GUI.
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _configure_project_imports() -> Path:
    b2_source_dir = _RL_ROOT / "source" / "B2_Lab"
    sys.path.insert(0, str(b2_source_dir))
    if args_cli.rsl_rl_type == "constraints":
        rsl_rl_source_dir = _RL_ROOT / "source" / "rsl_rl_constraints"
    else:
        rsl_rl_source_dir = _RL_ROOT / "source" / "rsl_rl_2.3.3"
    sys.path.insert(0, str(rsl_rl_source_dir))
    return rsl_rl_source_dir


_RSL_RL_SOURCE_DIR = _configure_project_imports()


import gymnasium as gym  # noqa: E402
import yaml  # noqa: E402

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils import class_to_dict  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import isaaclab_tasks  # noqa: E402, F401
import B2_Lab.tasks  # noqa: E402, F401
from B2_Lab.tasks.direct.b2_lab.agents.rsl_rl_ppo_cfg import B2LabFlatPPORunnerCfg  # noqa: E402
from B2_Lab.tasks.direct.b2_lab.agents.vecenv_wrapper import RslRlVecEnvWrapper  # noqa: E402

import rsl_rl  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


def _validate_rsl_rl_import() -> None:
    expected_dir = (_RSL_RL_SOURCE_DIR / "rsl_rl").resolve()
    loaded_path = Path(rsl_rl.__file__).resolve()
    if expected_dir not in loaded_path.parents:
        raise RuntimeError(
            f"Wrong rsl_rl package loaded: {loaded_path}. Expected a package under {expected_dir}."
        )


def _checkpoint_iteration(path: Path) -> int:
    match = _CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Checkpoint name must match model_<iteration>.pt: {path}")
    return int(match.group(1))


def _run_sort_key(run_dir: Path) -> tuple[datetime, int]:
    timestamp_text = run_dir.name[:19]
    try:
        timestamp = datetime.strptime(timestamp_text, "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        timestamp = datetime.fromtimestamp(run_dir.stat().st_mtime)
    return timestamp, run_dir.stat().st_mtime_ns


def _find_latest_checkpoint(log_root: Path) -> Path:
    if not log_root.is_dir():
        raise FileNotFoundError(f"Experiment log directory does not exist: {log_root}")

    runs: list[tuple[tuple[datetime, int], list[Path]]] = []
    for run_dir in log_root.iterdir():
        if not run_dir.is_dir():
            continue
        checkpoints = [
            path
            for path in run_dir.glob("model_*.pt")
            if _CHECKPOINT_PATTERN.fullmatch(path.name) is not None
        ]
        if checkpoints:
            runs.append((_run_sort_key(run_dir), checkpoints))

    if not runs:
        raise FileNotFoundError(f"No model_<iteration>.pt checkpoint found under {log_root}")

    _, newest_run_checkpoints = max(runs, key=lambda item: item[0])
    return max(newest_run_checkpoints, key=_checkpoint_iteration).resolve()


def _resolve_checkpoint(log_root: Path) -> Path:
    if args_cli.checkpoint is None:
        return _find_latest_checkpoint(log_root)

    requested = Path(args_cli.checkpoint).expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        candidates = [
            Path.cwd() / requested,
            _SCRIPT_DIR / requested,
            _RL_ROOT / requested,
        ]
    for candidate in candidates:
        if candidate.is_file():
            _checkpoint_iteration(candidate)
            return candidate.resolve()
    raise FileNotFoundError(f"Checkpoint file not found: {args_cli.checkpoint}")


def _resolve_output_dir(checkpoint: Path) -> Path:
    if args_cli.output_dir is None:
        run_dir = checkpoint.parent
        return run_dir / f"{run_dir.name}_export"
    output_dir = Path(args_cli.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    return output_dir.resolve()


class B2DeploymentPolicy(torch.nn.Module):
    """Deterministic actor wrapper: history -> CENet means -> action mean."""

    def __init__(
        self,
        actor: torch.nn.Module,
        auxiliary_networks: torch.nn.Module,
        frame_dim: int,
        history_length: int,
    ):
        super().__init__()
        self.actor = copy.deepcopy(actor.actor).cpu()
        self.cenet_encoder = copy.deepcopy(auxiliary_networks.cenet_encoder).cpu()
        self.cenet_mean_vel = copy.deepcopy(auxiliary_networks.cenet_mean_vel).cpu()
        self.cenet_mean_latent = copy.deepcopy(auxiliary_networks.cenet_mean_latent).cpu()
        self.frame_dim = int(frame_dim)
        self.history_length = int(history_length)

    def forward(self, prop_obs_history: torch.Tensor) -> torch.Tensor:
        newest_start = (self.history_length - 1) * self.frame_dim
        newest_end = self.history_length * self.frame_dim
        prop_obs = prop_obs_history[:, newest_start:newest_end]
        hidden = self.cenet_encoder(prop_obs_history)
        velocity = self.cenet_mean_vel(hidden)
        latent = self.cenet_mean_latent(hidden)
        actor_obs = torch.cat([prop_obs, velocity, latent], dim=-1)
        return self.actor(actor_obs)


def _yaml_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item) for item in value]
    if callable(value):
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", getattr(value, "__name__", str(value)))
        return f"{module}.{name}" if module else name
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _yaml_safe(item_method())
        except (TypeError, ValueError):
            pass
    return str(value)


def _dump_pickle(path: Path, config: object, fallback_dict: dict[str, Any]) -> None:
    try:
        with path.open("wb") as file:
            pickle.dump(config, file, protocol=pickle.HIGHEST_PROTOCOL)
    except (AttributeError, pickle.PicklingError, TypeError):
        with path.open("wb") as file:
            pickle.dump(fallback_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


def _make_deployment_config(raw_env: object, env_cfg: object, checkpoint: Path) -> dict[str, Any]:
    frame_dim = int(raw_env.cfg.num_proprio)
    history_length = int(raw_env.cfg.num_history_len)
    action_dim = int(raw_env.num_actions)
    input_dim = frame_dim * history_length
    if frame_dim != 49 or history_length != 10 or action_dim != 12:
        raise RuntimeError(
            "Unexpected B2 deployment dimensions: "
            f"frame={frame_dim}, history={history_length}, action={action_dim}"
        )

    joint_names = list(raw_env._robot.joint_names)
    if len(joint_names) != action_dim:
        raise RuntimeError(f"Joint name count mismatch: expected {action_dim}, got {len(joint_names)}")
    action_scale = raw_env._action_scale.detach().cpu().tolist()
    default_joint_position = raw_env._robot.data.default_joint_pos[0].detach().cpu().tolist()
    command_scale = raw_env.commands_scale.detach().cpu().tolist()
    step_dt = float(raw_env.step_dt)
    physics_dt = float(env_cfg.sim.dt)
    zero_threshold = float(raw_env.cfg.commands.zero_command_threshold)

    deployment = {
        "schema": "pkiro_b2_history_policy",
        "schema_version": 1,
        "source": {
            "checkpoint": str(checkpoint),
            "run": checkpoint.parent.name,
            "iteration": _checkpoint_iteration(checkpoint),
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "model": {
            "input_name": "prop_obs_history",
            "output_name": "actions",
            "input_dim": input_dim,
            "output_dim": action_dim,
        },
        "control": {
            "policy_dt": step_dt,
            "policy_hz": 1.0 / step_dt,
            "physics_dt": physics_dt,
            "physics_hz": 1.0 / physics_dt,
            "decimation": int(env_cfg.decimation),
        },
        "observation": {
            "frame_dim": frame_dim,
            "history_length": history_length,
            "history_order": "oldest_to_newest",
            "angular_velocity_scale": 0.25,
            "command_scale": command_scale,
            "joint_velocity_scale": 0.05,
            "layout": [
                {"name": "base_angular_velocity", "offset": 0, "size": 3},
                {"name": "projected_gravity", "offset": 3, "size": 3},
                {"name": "velocity_command", "offset": 6, "size": 3},
                {"name": "joint_position_relative", "offset": 9, "size": 12},
                {"name": "joint_velocity", "offset": 21, "size": 12},
                {"name": "previous_action", "offset": 33, "size": 12},
                {"name": "foot_clock", "offset": 45, "size": 4},
            ],
        },
        "action": {
            "dimension": action_dim,
            "joint_order": joint_names,
            "scale": action_scale,
            "default_joint_position": default_joint_position,
            "use_default_offset": True,
        },
        "gait": {
            "frequency_hz": 1.3,
            "phase_offsets": [0.0, 0.5, 0.5, 0.0],
            "foot_order": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            "stance_duration": 0.5,
            "zero_command_threshold": zero_threshold,
            "zero_hold_phase_threshold": 0.05,
        },
        "normalization": {
            "enabled": False,
            "mean_file": "obs_mean.txt",
            "std_file": "obs_std.txt",
        },
    }
    action_scale_map = {name: float(scale) for name, scale in zip(joint_names, action_scale)}
    return {
        "deployment": deployment,
        "actions": {
            "joint_pos": {
                "joint_order": joint_names,
                "scale": action_scale_map,
                "use_default_offset": True,
            }
        },
        "training_environment": _yaml_safe(class_to_dict(env_cfg)),
    }


def _write_stats(path: Path, value: float, count: int) -> None:
    lines = [f"{value:.18e}\n" for _ in range(count)]
    path.write_text("".join(lines), encoding="utf-8")


def _validate_torchscript(
    policy: B2DeploymentPolicy,
    scripted_policy: torch.jit.ScriptModule,
    input_dim: int,
) -> float:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    test_input = torch.randn(4, input_dim, generator=generator)
    with torch.inference_mode():
        expected = policy(test_input)
        actual = scripted_policy(test_input)
    max_error = float(torch.max(torch.abs(expected - actual)).item())
    if max_error > 1.0e-5:
        raise RuntimeError(f"TorchScript parity failed: max_abs_error={max_error}")
    return max_error


def _validate_onnx_if_available(
    onnx_path: Path,
    policy: B2DeploymentPolicy,
    input_dim: int,
) -> str:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        return "skipped (Python onnxruntime is unavailable; C++ runtime validation remains required)"

    generator = torch.Generator(device="cpu")
    generator.manual_seed(11)
    test_input = torch.randn(4, input_dim, generator=generator)
    with torch.inference_mode():
        expected = policy(test_input).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(["actions"], {"prop_obs_history": test_input.numpy()})[0]
    max_error = float(np.max(np.abs(expected - actual)))
    if max_error > 1.0e-5:
        raise RuntimeError(f"ONNX parity failed: max_abs_error={max_error}")
    return f"passed (max_abs_error={max_error:.3e})"


def _load_deployment_weights(runner: OnPolicyRunner, checkpoint: Path) -> None:
    checkpoint_data = torch.load(checkpoint, map_location=runner.device)
    runner.alg.actor.load_state_dict(checkpoint_data["model_state_dict"])
    auxiliary_result = runner.alg.aux_networks.load_state_dict(
        checkpoint_data["aux_net_state_dict"], strict=False
    )
    if auxiliary_result.missing_keys:
        raise RuntimeError(
            "Checkpoint is missing CENet weights required for deployment: "
            f"{auxiliary_result.missing_keys}"
        )
    if auxiliary_result.unexpected_keys:
        print(
            "[WARN] Ignoring unused legacy auxiliary weights: "
            f"{len(auxiliary_result.unexpected_keys)} tensors"
        )


def _export_bundle(
    runner: OnPolicyRunner,
    raw_env: object,
    env_cfg: object,
    agent_cfg: object,
    checkpoint: Path,
    output_dir: Path,
) -> None:
    deployment_config = _make_deployment_config(raw_env, env_cfg, checkpoint)
    deployment = deployment_config["deployment"]
    input_dim = int(deployment["model"]["input_dim"])
    output_dim = int(deployment["model"]["output_dim"])
    policy = B2DeploymentPolicy(
        runner.alg.actor,
        runner.alg.aux_networks,
        int(deployment["observation"]["frame_dim"]),
        int(deployment["observation"]["history_length"]),
    ).eval()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".deploy_b2_", dir=output_dir.parent) as temp_dir_name:
        stage_dir = Path(temp_dir_name)
        params_dir = stage_dir / "params"
        params_dir.mkdir(parents=True)

        example_input = torch.zeros(1, input_dim)
        with torch.inference_mode():
            example_output = policy(example_input)
        if tuple(example_output.shape) != (1, output_dim):
            raise RuntimeError(
                f"Policy output shape mismatch: expected (1, {output_dim}), got {tuple(example_output.shape)}"
            )

        scripted_policy = torch.jit.trace(policy, example_input, strict=True)
        scripted_policy = torch.jit.freeze(scripted_policy.eval())
        scripted_policy.save(str(stage_dir / "policy.pt"))
        jit_error = _validate_torchscript(policy, scripted_policy, input_dim)

        torch.onnx.export(
            policy,
            example_input,
            stage_dir / "policy.onnx",
            export_params=True,
            opset_version=args_cli.opset,
            do_constant_folding=True,
            input_names=["prop_obs_history"],
            output_names=["actions"],
            dynamic_axes={
                "prop_obs_history": {0: "batch_size"},
                "actions": {0: "batch_size"},
            },
            verbose=False,
        )
        onnx_status = _validate_onnx_if_available(stage_dir / "policy.onnx", policy, input_dim)

        _write_stats(stage_dir / "obs_mean.txt", 0.0, input_dim)
        _write_stats(stage_dir / "obs_std.txt", 1.0, input_dim)
        with (params_dir / "env.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(deployment_config, file, sort_keys=False, allow_unicode=True)

        agent_dict = _yaml_safe(agent_cfg.to_dict())
        with (params_dir / "agent.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(agent_dict, file, sort_keys=False, allow_unicode=True)
        _dump_pickle(params_dir / "env.pkl", env_cfg, deployment_config)
        _dump_pickle(params_dir / "agent.pkl", agent_cfg, agent_dict)

        required_files = [
            Path("policy.pt"),
            Path("policy.onnx"),
            Path("obs_mean.txt"),
            Path("obs_std.txt"),
            Path("params/env.yaml"),
            Path("params/env.pkl"),
            Path("params/agent.yaml"),
            Path("params/agent.pkl"),
        ]
        output_dir.mkdir(parents=True, exist_ok=True)
        for relative_path in required_files:
            source = stage_dir / relative_path
            if not source.is_file() or source.stat().st_size == 0:
                raise RuntimeError(f"Deployment artifact is missing or empty: {relative_path}")
        for relative_path in required_files:
            target = output_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage_dir / relative_path, target)

    print(f"[INFO] Checkpoint: {checkpoint}")
    print(f"[INFO] Deployment bundle: {output_dir}")
    print(f"[INFO] TorchScript parity: passed (max_abs_error={jit_error:.3e})")
    print(f"[INFO] ONNX parity: {onnx_status}")
    print(f"[INFO] Policy contract: input=[batch,{input_dim}], output=[batch,{output_dim}]")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: B2LabFlatPPORunnerCfg,
) -> None:
    _validate_rsl_rl_import()
    experiment_name = _DEFAULT_EXPERIMENT
    if args_cli.experiment_name is not None:
        experiment_name = args_cli.experiment_name
    log_root = _DEFAULT_LOG_ROOT / experiment_name
    checkpoint = _resolve_checkpoint(log_root)
    output_dir = _resolve_output_dir(checkpoint)

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.experiment_name = experiment_name
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    try:
        if isinstance(env.unwrapped, DirectMARLEnv):
            raise RuntimeError("B2 deployment expects a single-agent environment")
        raw_env = env.unwrapped
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        _load_deployment_weights(runner, checkpoint)
        runner.eval_mode()
        _export_bundle(runner, raw_env, env_cfg, agent_cfg, checkpoint, output_dir)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
