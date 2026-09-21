import gymnasium as gym  # noqa: F401

gym.register(
    id="B2",
    entry_point="B2_Lab.tasks.direct.b2_lab.b2_lab_env:B2LabEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "B2_Lab.tasks.direct.b2_lab.b2_lab_env_cfg:B2LabFlatEnvCfg",
        "rsl_rl_cfg_entry_point": "B2_Lab.tasks.direct.b2_lab.agents.rsl_rl_ppo_cfg:B2LabFlatPPORunnerCfg",
    },
)

# Extreme Parkour teacher/student task (uses rsl_rl_2.3.3 + ActorCriticParkour).
gym.register(
    id="B2-Parkour",
    entry_point="B2_Lab.tasks.direct.b2_lab.b2_lab_parkour_env:B2LabParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "B2_Lab.tasks.direct.b2_lab.b2_lab_parkour_env_cfg:B2LabParkourEnvCfg",
        "rsl_rl_cfg_entry_point": "B2_Lab.tasks.direct.b2_lab.agents.rsl_rl_ppo_cfg:B2LabParkourPPORunnerCfg",
    },
)
