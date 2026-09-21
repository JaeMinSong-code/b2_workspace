"""play_b2.py 재생 데이터 로깅 + 종료 시 PNG 저장.

실시간 플롯이 아니라, 재생 루프 동안 값만 쌓아뒀다가 종료 시점에
한 번에 그려서 PNG로 저장한다.

주의: 이 파일 이름을 logging.py로 바꾸면 파이썬 표준 logging 모듈을
가려서 isaaclab 내부 import가 깨진다. play_logger 이름을 유지할 것.
"""

import os
import time

import matplotlib

matplotlib.use("Agg")  # 창 없이 파일로만 저장 (headless 안전)
import matplotlib.pyplot as plt
import numpy as np

FOOT_NAMES = ["FL", "FR", "RL", "RR"]


def _quat_to_roll_pitch(q) -> tuple[float, float]:
    """(w, x, y, z) 쿼터니언 → roll, pitch [rad]."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


class PlayLogger:
    """B2LabEnv에서 매 스텝 상태를 기록하고 종료 시 PNG로 플롯."""

    def __init__(self, env, env_idx: int = 0):
        self.env = env  # B2LabEnv (unwrapped)
        self.idx = env_idx
        self.joint_names = list(env._robot.data.joint_names)  # [12] 관절 이름
        self.t = []
        self.cmd = []           # [vx, vy, wz] reference
        self.lin_vel = []       # body frame [vx, vy] actual
        self.ang_vel_z = []     # actual yaw rate
        self.roll_pitch = []    # [roll, pitch] (rad)
        self.body_height = []   # 지형 기준 몸통 높이 (height scanner)
        self.contact_des = []   # [4] desired contact (0~1, gait 위상 smoothing)
        self.contact_act = []   # [4] actual contact (0/1)
        self.foot_height = []   # [4] 발 world z
        self.joint_pos = []     # [12] 관절 각도 (rad)
        self.joint_vel = []     # [12] 관절 각속도 (rad/s)
        self.joint_torque = []  # [12] 인가 토크 (Nm, applied_torque = 클리핑 후)
        self.actions = []       # [num_actions] 정책 출력 액션 (있을 때만)
        self.prop_obs = []      # [num_proprio] 현재 proprioceptive observation (있을 때만)

    def log_step(self, sim_time: float, prop_obs=None, actions=None):
        """한 스텝 기록. prop_obs/actions 는 [B, D] 텐서(주면 obs/action 도 로깅)."""
        env, i = self.env, self.idx
        robot = env._robot
        self.t.append(sim_time)
        self.cmd.append(env._commands[i, :3].detach().cpu().numpy().copy())
        self.lin_vel.append(robot.data.root_lin_vel_b[i, :2].detach().cpu().numpy().copy())
        self.ang_vel_z.append(float(robot.data.root_ang_vel_b[i, 2]))
        roll, pitch = _quat_to_roll_pitch(robot.data.root_quat_w[i].detach().cpu().numpy())
        self.roll_pitch.append([roll, pitch])
        self.body_height.append(float(env.com_height[i, 0]))
        self.contact_des.append(env.desired_contact_states[i].detach().cpu().numpy().copy())
        self.contact_act.append(env.foot_contact_state[i].float().detach().cpu().numpy().copy())
        self.foot_height.append(env.foot_positions[i, :, 2].detach().cpu().numpy().copy())
        self.joint_pos.append(robot.data.joint_pos[i].detach().cpu().numpy().copy())
        self.joint_vel.append(robot.data.joint_vel[i].detach().cpu().numpy().copy())
        self.joint_torque.append(robot.data.applied_torque[i].detach().cpu().numpy().copy())
        if actions is not None:
            self.actions.append(actions[i].detach().cpu().numpy().copy())
        if prop_obs is not None:
            self.prop_obs.append(prop_obs[i].detach().cpu().numpy().copy())

    def save(self, out_dir: str, target_height: float | None = None) -> str | None:
        """플롯을 항목별로 각각 PNG 한 장씩, 한 폴더 안에 저장. 폴더 경로 반환.

        모든 그래프는 선(line)만 사용 — 색 채우기 없음.
        데이터가 없으면 None.
        """
        if len(self.t) < 2:
            return None

        t = np.asarray(self.t)
        cmd = np.asarray(self.cmd)                 # [T, 3]
        lin_vel = np.asarray(self.lin_vel)         # [T, 2]
        ang_vel_z = np.asarray(self.ang_vel_z)     # [T]
        roll_pitch = np.rad2deg(np.asarray(self.roll_pitch))  # [T, 2] deg
        body_height = np.asarray(self.body_height)  # [T]
        contact_des = np.asarray(self.contact_des)  # [T, 4]
        contact_act = np.asarray(self.contact_act)  # [T, 4]
        foot_height = np.asarray(self.foot_height)  # [T, 4]

        # 이번 재생 결과를 담을 폴더 (타임스탬프)
        run_dir = os.path.join(out_dir, time.strftime("play_%Y-%m-%d_%H-%M-%S"))
        os.makedirs(run_dir, exist_ok=True)

        def _save(name: str, plot_fn, ylabel: str, title: str):
            """단일 항목 플롯 한 장 저장."""
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_fn(ax)
            ax.set_xlabel("time [s]")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, lw=0.3, alpha=0.5)
            ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(os.path.join(run_dir, name), dpi=150)
            plt.close(fig)

        # 1) 몸통 선속도 vx (ref vs actual)
        def _vx(ax):
            ax.plot(t, cmd[:, 0], "k--", label="ref")
            ax.plot(t, lin_vel[:, 0], "b-", label="actual")
        _save("01_lin_vel_x.png", _vx, "vx [m/s]", "Body linear velocity x")

        # 2) 몸통 선속도 vy (ref vs actual)
        def _vy(ax):
            ax.plot(t, cmd[:, 1], "k--", label="ref")
            ax.plot(t, lin_vel[:, 1], "b-", label="actual")
        _save("02_lin_vel_y.png", _vy, "vy [m/s]", "Body linear velocity y")

        # 3) 몸통 각속도 wz (ref vs actual)
        def _wz(ax):
            ax.plot(t, cmd[:, 2], "k--", label="ref")
            ax.plot(t, ang_vel_z, "b-", label="actual")
        _save("03_ang_vel_z.png", _wz, "wz [rad/s]", "Body angular velocity (yaw)")

        # 4) 자세 (roll, pitch)
        def _ori(ax):
            ax.plot(t, roll_pitch[:, 0], "b-", label="roll")
            ax.plot(t, roll_pitch[:, 1], "r-", label="pitch")
            ax.axhline(0.0, color="k", lw=0.5)
        _save("04_orientation.png", _ori, "angle [deg]", "Body orientation (roll/pitch)")

        # 5) 몸통 높이
        def _bh(ax):
            ax.plot(t, body_height, "b-", label="actual")
            if target_height is not None:
                ax.axhline(target_height, color="k", ls="--", label=f"target {target_height:.2f}")
        _save("05_body_height.png", _bh, "height [m]", "Body height")

        # 6) 발별 contact (desired vs actual) — 발마다 개별 PNG
        for f in range(4):
            def _ct(ax, f=f):
                ax.plot(t, contact_des[:, f], "k--", lw=1.0, label="desired")
                ax.plot(t, contact_act[:, f], "b-", lw=1.0, label="actual")
                ax.set_ylim(-0.05, 1.05)
            _save(f"06_contact_{FOOT_NAMES[f]}.png", _ct,
                  "contact (0/1)", f"Foot contact {FOOT_NAMES[f]}")

        # 7) 발 높이 (world z) — 발마다 개별 PNG
        for f in range(4):
            def _fh(ax, f=f):
                ax.plot(t, foot_height[:, f], "b-", label=FOOT_NAMES[f])
            _save(f"07_foot_height_{FOOT_NAMES[f]}.png", _fh,
                  "foot z [m]", f"Foot height {FOOT_NAMES[f]}")

        # 8~10) 관절 각도/각속도/토크 — 다리별로 묶어서 (다리당 hip/thigh/calf 3선)
        joint_pos = np.asarray(self.joint_pos)        # [T, 12]
        joint_vel = np.asarray(self.joint_vel)        # [T, 12]
        joint_torque = np.asarray(self.joint_torque)  # [T, 12]
        names = self.joint_names

        # 관절 이름 접두사(FL/FR/RL/RR)로 인덱스 그룹핑. 못 맞추면 전체를 한 그룹으로.
        leg_groups: dict[str, list[int]] = {}
        for j, nm in enumerate(names):
            leg = nm.split("_")[0]
            leg_groups.setdefault(leg, []).append(j)

        def _joint_plot(prefix: str, data: np.ndarray, ylabel: str, title: str):
            for leg, idxs in leg_groups.items():
                def _p(ax, idxs=idxs):
                    for j in idxs:
                        ax.plot(t, data[:, j], "-", lw=1.0, label=names[j])
                _save(f"{prefix}_{leg}.png", _p, ylabel, f"{title} — {leg}")

        _joint_plot("08_joint_pos", joint_pos, "angle [rad]", "Joint position")
        _joint_plot("09_joint_vel", joint_vel, "vel [rad/s]", "Joint velocity")
        _joint_plot("10_joint_torque", joint_torque, "torque [Nm]", "Joint torque")

        # 그래프에 쓴 값 + observation/action 을 같은 폴더에 log.csv 로 남긴다.
        self._save_csv(
            run_dir, t, cmd, lin_vel, ang_vel_z, roll_pitch, body_height,
            contact_des, contact_act, foot_height, joint_pos, joint_vel, joint_torque,
        )
        return run_dir

    def _save_csv(self, run_dir, t, cmd, lin_vel, ang_vel_z, roll_pitch, body_height,
                  contact_des, contact_act, foot_height, joint_pos, joint_vel, joint_torque):
        """플롯에 사용한 값(+ observation/action)을 한 행/스텝으로 평탄화해 log.csv 저장.

        orientation 은 플롯과 동일하게 deg 단위. contact 는 desired/actual 둘 다.
        actions/prop_obs 는 log_step 에 넘겼을 때만 열이 추가된다.
        """
        cols: list[str] = []
        data_cols: list[np.ndarray] = []

        def add(name, arr, labels=None):
            arr = np.asarray(arr)
            if arr.ndim == 1:
                cols.append(name)
                data_cols.append(arr.reshape(-1, 1))
            else:
                n = arr.shape[1]
                lab = labels if labels is not None else [str(k) for k in range(n)]
                cols.extend(f"{name}_{lab[k]}" for k in range(n))
                data_cols.append(arr)

        jn = self.joint_names
        add("t", t)
        add("cmd", cmd, ["vx", "vy", "wz"])
        add("lin_vel", lin_vel, ["x", "y"])
        add("ang_vel_z", ang_vel_z)
        add("orientation_deg", roll_pitch, ["roll", "pitch"])
        add("body_height", body_height)
        add("contact_des", contact_des, FOOT_NAMES)
        add("contact_act", contact_act, FOOT_NAMES)
        add("foot_height", foot_height, FOOT_NAMES)
        add("joint_pos", joint_pos, jn)
        add("joint_vel", joint_vel, jn)
        add("joint_torque", joint_torque, jn)
        if self.actions:
            act = np.asarray(self.actions)
            act_lab = [jn[k] if k < len(jn) else str(k) for k in range(act.shape[1])]
            add("action", act, act_lab)
        if self.prop_obs:
            add("obs", np.asarray(self.prop_obs))

        table = np.concatenate(data_cols, axis=1)
        path = os.path.join(run_dir, "log.csv")
        np.savetxt(path, table, delimiter=",", header=",".join(cols), comments="", fmt="%.6g")
        print(f"[PlayLogger] log.csv 저장: {path} ({table.shape[0]}행 {table.shape[1]}열)")
