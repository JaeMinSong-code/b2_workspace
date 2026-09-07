import argparse
import os
import glob
import yaml
import torch

from rsl_rl.runners import OnPolicyRunner


# -----------------------------
# cfg 파일 자동 탐색/로드
# -----------------------------
def find_cfg_file(run_dir: str) -> str:
    candidates = [
        os.path.join(run_dir, "train_cfg.yaml"),
        os.path.join(run_dir, "train_cfg.yml"),
        os.path.join(run_dir, "config.yaml"),
        os.path.join(run_dir, "config.yml"),
        os.path.join(run_dir, ".hydra", "config.yaml"),
        os.path.join(run_dir, ".hydra", "config.yml"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    # 그래도 없으면 run_dir 아래 yaml 전부 중에서 'algorithm' 키가 있는 걸 우선
    yamls = glob.glob(os.path.join(run_dir, "**", "*.yaml"), recursive=True) + \
            glob.glob(os.path.join(run_dir, "**", "*.yml"), recursive=True)
    for p in yamls:
        try:
            with open(p, "r") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and ("algorithm" in data or "policy" in data):
                return p
        except Exception:
            pass

    raise FileNotFoundError(
        f"Could not find runner/train cfg yaml under: {run_dir}\n"
        f"Tried: {candidates}"
    )


def load_train_cfg_from_yaml(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"cfg is not a dict: {cfg_path}")

    # rsl_rl OnPolicyRunner expects the "train_cfg" dict that contains algorithm/policy/runner/etc.
    # 어떤 로그는 최상위에 바로 있고, 어떤 건 nested일 수 있어요.
    # 아래는 흔한 케이스들을 커버:
    if "runner" in cfg and "algorithm" in cfg and "policy" in cfg:
        return cfg
    if "train_cfg" in cfg and isinstance(cfg["train_cfg"], dict):
        return cfg["train_cfg"]

    # hydra 전체 config에서 rsl_rl 부분만 들어있는 케이스(프로젝트마다 다름)
    # 최대한 "algorithm/policy" 세트가 보이는 dict를 찾아 반환
    def search_dict(d):
        if isinstance(d, dict):
            if "algorithm" in d and "policy" in d:
                return d
            for v in d.values():
                out = search_dict(v)
                if out is not None:
                    return out
        return None

    found = search_dict(cfg)
    if found is not None:
        return found

    raise KeyError(
        f"Could not locate train cfg dict with keys like ['algorithm','policy'] in: {cfg_path}\n"
        f"Top-level keys={list(cfg.keys())}"
    )


# -----------------------------
# checkpoint 로드 (분리 키 대응)
# -----------------------------
def load_ckpt(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint is not a dict.")
    # 여기서는 train_cfg가 ckpt에 없으니 state_dict만 반환
    return ckpt


# -----------------------------
# dims 추정 (aux_net_state_dict 키에 맞춤)
# -----------------------------
def infer_dims_from_state_dicts(model_sd: dict, aux_sd: dict):
    # actor input dim
    actor_in = None
    for k, v in model_sd.items():
        if "actor" in k and k.endswith(".weight") and isinstance(v, torch.Tensor) and v.ndim == 2:
            actor_in = int(v.shape[1])
            break
    if actor_in is None:
        raise RuntimeError("Could not infer actor input dim (no actor.*.weight).")

    # action dim: actor bias 중 작은 것
    actor_biases = [
        v.numel()
        for k, v in model_sd.items()
        if "actor" in k and k.endswith(".bias") and isinstance(v, torch.Tensor) and v.ndim == 1
    ]
    if not actor_biases:
        raise RuntimeError("Could not infer action dim (no actor.*.bias).")
    act_dim = int(min(actor_biases))

    # v/z dim: aux 네이밍은 프로젝트마다 다르므로 최대한 유연하게
    # vel logvar bias 후보
    v_key = None
    for cand in ["cenet_logvar_vel.bias", "logvar_vel.bias", "vel_logvar.bias"]:
        if cand in aux_sd and aux_sd[cand].ndim == 1:
            v_key = cand
            break
    if v_key is None:
        # fallback: 이름에 'vel'+'logvar'+'bias'가 들어간 1D 텐서
        for k, v in aux_sd.items():
            if ("vel" in k) and ("logvar" in k) and k.endswith("bias") and isinstance(v, torch.Tensor) and v.ndim == 1:
                v_key = k
                break
    if v_key is None:
        raise RuntimeError("Could not infer v_dim from aux_net_state_dict (no vel logvar bias found).")
    v_dim = int(aux_sd[v_key].numel())

    # z/priv logvar bias 후보
    z_key = None
    for cand in ["cenet_logvar_priv.bias", "cenet_logvar_z.bias", "logvar_priv.bias", "logvar_z.bias"]:
        if cand in aux_sd and aux_sd[cand].ndim == 1:
            z_key = cand
            break
    if z_key is None:
        # fallback: vel 아닌 logvar bias 중 하나
        for k, v in aux_sd.items():
            if ("logvar" in k) and ("vel" not in k) and k.endswith("bias") and isinstance(v, torch.Tensor) and v.ndim == 1:
                z_key = k
                break
    if z_key is None:
        raise RuntimeError("Could not infer z_dim from aux_net_state_dict (no non-vel logvar bias found).")
    z_dim = int(aux_sd[z_key].numel())

    P = actor_in - v_dim - z_dim
    if P <= 0:
        raise RuntimeError(f"Bad inferred P={P} (actor_in={actor_in}, v_dim={v_dim}, z_dim={z_dim})")

    # history input: aux의 cenet*.weight 중 in_features 최대
    hist_in = None
    for k, v in aux_sd.items():
        if "cenet" in k and k.endswith(".weight") and isinstance(v, torch.Tensor) and v.ndim == 2:
            hist_in = int(v.shape[1]) if hist_in is None else max(hist_in, int(v.shape[1]))
    if hist_in is None:
        # fallback: weight 중 in_features 최대
        for k, v in aux_sd.items():
            if k.endswith(".weight") and isinstance(v, torch.Tensor) and v.ndim == 2:
                hist_in = int(v.shape[1]) if hist_in is None else max(hist_in, int(v.shape[1]))
    if hist_in is None:
        raise RuntimeError("Could not infer history input size from aux_net_state_dict (no 2D weights).")

    if hist_in % P != 0:
        raise RuntimeError(f"hist_in({hist_in}) not divisible by P({P})")
    H = hist_in // P

    return P, H, act_dim


# -----------------------------
# 최소 DummyEnv
# -----------------------------
class _DummyCfg:
    def __init__(self, num_proprio: int, num_history_len: int):
        self.num_proprio = int(num_proprio)
        self.num_history_len = int(num_history_len)


class DummyEnv:
    def __init__(self, actor_obs_dim: int, critic_obs_dim: int, num_actions: int, P: int, H: int, device: str):
        self.device = device
        self.num_envs = 1
        self.num_actions = int(num_actions)
        self.cfg = _DummyCfg(P, H)
        self._obs = {
            "observations": {
                "policy": torch.zeros(1, int(actor_obs_dim), device=self.device),
                "critic": torch.zeros(1, int(critic_obs_dim), device=self.device),
            }
        }

    def get_observations(self):
        return None, self._obs

    def reset(self):
        return self.get_observations()

    def step(self, actions):
        return None, None, None, None, self._obs


# -----------------------------
# ONNX export
# -----------------------------
def export_onnx(runner: OnPolicyRunner, P: int, H: int, out_path: str, opset: int):
    class PolicyWrapper(torch.nn.Module):
        def __init__(self, actor, aux, P: int, H: int):
            super().__init__()
            self.actor = actor
            self.aux = aux
            self.P = int(P)
            self.H = int(H)

        def forward(self, prop_obs_history: torch.Tensor):
            prop_obs = prop_obs_history[:, (self.H - 1) * self.P : self.H * self.P]
            v_hat, z_hat = self.aux.cenet_infer(prop_obs_history)
            actor_obs = torch.cat([prop_obs, v_hat, z_hat], dim=-1)
            return self.actor(actor_obs)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    wrapped = PolicyWrapper(runner.alg.actor, runner.alg.aux_networks, P, H).eval()
    dummy_in = torch.randn(1, P * H, device=runner.device)

    torch.onnx.export(
        wrapped, dummy_in, out_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["prop_obs_history"],
        output_names=["actions"],
        dynamic_axes={"prop_obs_history": {0: "batch_size"}, "actions": {0: "batch_size"}},
        verbose=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--opset", type=int, default=11)
    args = ap.parse_args()

    ckpt_path = os.path.abspath(args.checkpoint)
    run_dir = os.path.dirname(ckpt_path)

    # 1) cfg yaml 찾기/로드
    cfg_path = find_cfg_file(run_dir)
    train_cfg = load_train_cfg_from_yaml(cfg_path)

    # 2) ckpt 로드
    ckpt = load_ckpt(ckpt_path)

    # 3) state dict 가져오기 (당신 체크포인트 키에 맞춤)
    model_sd = ckpt["model_state_dict"]
    aux_sd = ckpt["aux_net_state_dict"]

    # 4) dim 추정
    P, H, act_dim = infer_dims_from_state_dicts(model_sd, aux_sd)
    # actor input dim (wrapper가 concat하는 actor_obs dim과 일치해야 함)
    actor_in = next(int(v.shape[1]) for k, v in model_sd.items()
                    if "actor" in k and k.endswith(".weight") and v.ndim == 2)
    critic_in = actor_in  # runner init 용. 필요하면 critic weight에서 읽어도 됨.

    print(f"[INFO] checkpoint: {ckpt_path}")
    print(f"[INFO] cfg yaml : {cfg_path}")
    print(f"[INFO] inferred: P={P}, H={H}, act_dim={act_dim}, actor_in={actor_in}")

    # 5) runner 생성 + 분리 state_dict 직접 로드
    env = DummyEnv(actor_in, critic_in, act_dim, P, H, device=args.device)
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=args.device)

    # 주의: alg 내부 이름은 프로젝트 구현에 따라 다를 수 있어요.
    runner.alg.actor_critic.load_state_dict(model_sd, strict=False)

    # reward critic / cost critic / aux
    if hasattr(runner.alg, "reward_critic"):
        runner.alg.reward_critic.load_state_dict(ckpt["reward_critic_state_dict"], strict=False)
    if hasattr(runner.alg, "cost_critic"):
        runner.alg.cost_critic.load_state_dict(ckpt["cost_critic_model_state_dict"], strict=False)
    if hasattr(runner.alg, "aux_networks"):
        runner.alg.aux_networks.load_state_dict(aux_sd, strict=False)

    # 6) export 경로
    out_path = os.path.abspath(args.out) if args.out else os.path.join(run_dir, "policy.onnx")
    export_onnx(runner, P, H, out_path, args.opset)
    print(f"[INFO] exported: {out_path}")


if __name__ == "__main__":
    main()
