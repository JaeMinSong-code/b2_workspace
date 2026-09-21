# Extreme Parkour on visionRL_ws_50hz — implementation notes

Implements the two-phase **Extreme Parkour** pipeline (Cheng et al., ICRA 2024)
on the B2 quadruped (IsaacLab direct workflow + rsl_rl_2.3.3). The existing
velocity-tracking / CENet pipeline is left fully intact — everything here is
additive (new files + subclasses + a couple of tiny additive edits).

> Status: code is written and the **rsl_rl modules are numerically verified**
> (forward passes, shapes, PPO interface). The IsaacLab env pieces compile but
> are **not yet validated in sim** (no GPU/sim here). See "Validation checklist".

---

## 1. The two phases

**Phase 1 — teacher (privileged, scandots).** An MLP policy that observes
proprioception + a heightmap scandots scan + privileged latents, trained with
**PPO**. Encoders: scandots encoder, privileged encoder, and a history/RMA
adaptation encoder. Reward = goal-velocity tracking + yaw tracking + a set of
regularizers (see §4).

**Phase 2 — student (depth vision).** A CNN+GRU depth backbone predicts the
teacher's scandots latent (and the yaw) from a single forward depth image; the
student actor is distilled to match the teacher's actions. This is the
teacher→student distillation from the paper.

---

## 2. Observation contract (the "policy" vector, 675-dim)

The env packs a single flat vector (Extreme-Parkour layout):

```
[ proprio(49) | scandots(121) | priv_explicit(3) | priv_latent(33) | history(10*49=490) ]  = 696
```
(scandots = 11x11 height-scan grid, not 10x10 — GridPattern includes endpoints.)

- **proprio(49)**: `[ang_vel*0.25 (3), projected_gravity (3), (cmd_vel, delta_yaw, delta_next_yaw) (3),
  joint_pos-default (12), joint_vel*0.05 (12), last_action (12), gait_clock (4)]`.
  The depth backbone predicts `(delta_yaw, delta_next_yaw)` and overwrites indices **[7:9]**.
- **scandots(100)**: existing 10×10 `height_scanner` ray grid (ground-relative, clipped).
- **priv_explicit(3)**: base linear velocity (`root_lin_vel_b*2`).
- **priv_latent(33)**: existing `privileged_obs_buf` (foot contact, friction, actuator gains, CoM height).
- **history(490)**: 10-step history of the parkour proprio (its own buffer, separate from the base env's).

`ActorCriticParkour` slices this internally and encodes scandots/priv/history.
Critic uses the same 675 vector (symmetric privileged critic).

---

## 3. Files

> **Runner bugfix (important):** `runners/on_policy_runner.py` rollout loop did
> not update `actor_obs`/`critic_obs` after `env.step` (it computed a normalized
> `obs` and discarded it), so PPO acted on the initial observation for the whole
> rollout and bootstrapped returns incorrectly. Fixed to reassign both each step.
> Only affects the `original` (rsl_rl_2.3.3) path used by parkour.

### New — rsl_rl_2.3.3 (the "learning structure")
- `source/rsl_rl_2.3.3/rsl_rl/modules/actor_critic_parkour.py`
  - `ActorCriticParkour` (teacher; PPO-compatible policy) + `StateHistoryEncoder` (RMA).
- `source/rsl_rl_2.3.3/rsl_rl/modules/depth_backbone.py`
  - `DepthOnlyFCBackbone` (CNN) + `RecurrentDepthBackbone` (CNN+GRU, outputs scan latent + yaw).
- Edited `modules/__init__.py` and `runners/on_policy_runner.py` (import + register the class
  so the runner's `eval(class_name)` finds `ActorCriticParkour`).

### New — B2_Lab (depth + reward + env)
- `.../b2_lab/b2quad_parkour_reward.py` — `B2quadParkourReward` (14 EP reward terms).
- `.../b2_lab/b2_lab_parkour_env_cfg.py` — `B2LabParkourEnvCfg` (depth camera, obs dims, EP reward scales).
- `.../b2_lab/b2_lab_parkour_env.py` — `B2LabParkourEnv` (depth camera, goal/yaw, parkour obs assembly).
- Edited `b2_lab_env.py::_prepare_reward_function` — register the parkour reward container (additive).
- Edited `agents/rsl_rl_ppo_cfg.py` — `ParkourActorCriticCfg` + `B2LabParkourPPORunnerCfg`.
- Edited `tasks/direct/__init__.py` — register gym task **`B2-Parkour`**.

### New / edited — scripts
- Edited `scripts/rsl_rl/train_b2.py` — `--parkour` flag swaps env+agent cfg and forces `--rsl_rl_type original`.
- New `scripts/rsl_rl/train_parkour_vision.py` — Phase-2 depth distillation.

---

## 4. Rewards (weights = reference EP, per second; env multiplies by dt)

| term | weight | note |
|---|---|---|
| tracking_goal_vel | +1.5 | velocity projected on goal dir, capped at cmd_vel |
| tracking_yaw | +0.5 | exp(-|delta_yaw|) |
| lin_vel_z | -1.0 | |
| ang_vel_xy | -0.05 | |
| orientation | -1.0 | |
| dof_acc | -2.5e-7 | |
| torques | -1e-5 | |
| delta_torques | -1e-7 | uses `last_torques` |
| action_rate | -0.1 | L2 norm of action delta |
| hip_pos | -0.5 | hip joint deviation |
| dof_error | -0.04 | all-joint deviation |
| collision | -10.0 | contact on base/thigh/calf |
| feet_stumble | -1.0 | horizontal >> vertical foot force |
| feet_edge | -1.0 | **inactive** until terrain exposes `env.feet_at_edge` |
| termination | -1.0 | non-timeout resets |

`tracking_sigma = 0.2`.

---

## 5. How to run

**Phase 1 (teacher):**
```bash
cd visionRL_ws_50hz/scripts/rsl_rl
python train_b2.py --parkour --num_envs 4096
# -> logs/rsl_rl/b2_original/<date>/  (experiment_name b2_parkour)
```

**Phase 2 (vision student):**
```bash
python train_parkour_vision.py --num_envs 256 --enable_cameras \
    --teacher_ckpt /abs/path/to/phase1/model_XXXX.pt
```

Both use `rsl_rl_2.3.3`. Phase 2 needs `--enable_cameras` (RTX depth sensor).

---

## 6. Status of features (updated after sim validation)

1. **Goal system = world-space forward waypoints (DONE).** Each env lays out
   `num_goals` waypoints along +x from spawn; `target_yaw`/`delta_yaw`/`goal_dir`
   are computed from the actual next goal, and the robot advances goals as it
   reaches them (`cfg.goals`). Works on plane and obstacle terrain.
   *Remaining refinement:* goals are a straight forward line, not aligned to
   specific obstacle geometry (would need custom terrain functions that export
   per-obstacle goal positions).
2. **RMA history encoder is now trained in Phase 1 (DONE).** PPO adds an
   `adaptation` loss (regress history latent → detached priv latent), gated by
   `hasattr(policy, "adaptation_loss")` so vanilla ActorCritic is unaffected.
   Verified decreasing in sim (0.043 → 0.002). Phase 2 can run `--hist_encoding`
   for a deployable (priv-free) student.
3. **`feet_edge` is active (DONE).** Detected from the per-ray height spread of the
   foot scanner (`cfg.goals.edge_threshold`); ~0 on plane, non-zero on obstacle
   terrain (verified -0.026).
4. **Depth camera intrinsics/mount** (`focal_length`, `horizontal_aperture`, `rot`,
   position) are reasonable defaults — still **tune to the real B2 RealSense mount**.
5. Costs (B2quadCost) still compute but are **ignored** by PPO. Harmless; can be
   disabled for speed.

Obstacle-terrain training: `train_b2.py --parkour --terrain` (uses
`B2LabParkourTerrainEnvCfg`, curriculum stairs/boxes/slopes). Plane is the default
`--parkour` for fast iteration.

---

## 7. Validation (DONE in sim — env `env_parkour_50gpu`, RTX 5090, IsaacSim 5.1)

Conda env: cloned `env_isaaclab_50gpu` → `env_parkour_50gpu`, then
`pip install -e source/B2_Lab --no-deps` (repoints B2_Lab to visionRL_ws_50hz).

- [x] `train_b2.py --parkour --num_envs 512` runs; builds `ActorCriticParkour`
      (`actor_in=104, critic_in=696`); PPO losses print; ~2000 steps/s.
- [x] Reward signal healthy & correct sign: `tracking_goal_vel`/`tracking_yaw`
      positive, `collision`/`orientation` negative.
- [x] RMA `adaptation` loss trains and decreases (0.043 → 0.002).
- [x] `--terrain` obstacle variant runs; `feet_edge` activates (−0.026);
      `Curriculum/terrain_level` logged.
- [x] Phase-2 `train_parkour_vision.py` runs end-to-end from a teacher ckpt:
      depth camera renders, `action/scan/yaw` distillation losses compute, env
      steps with student actions, GRU hidden resets.
- [x] `@configclass` accepts the list defaults in `ParkourActorCriticCfg`.

### Bugs found & fixed during validation
- `num_scan` was 100 but the height scanner yields **121** (11x11). Fixed in cfg +
  policy cfg.
- Runner rollout never updated `actor_obs`/`critic_obs` after `env.step` → fixed.

### Not yet exercised
- Long training to convergence (only short smoke runs).
- Phase-2 `--hist_encoding` deployable path end-to-end (the history encoder itself
  is validated numerically + trained via RMA; the flag just swaps which encoder
  feeds the actor).
