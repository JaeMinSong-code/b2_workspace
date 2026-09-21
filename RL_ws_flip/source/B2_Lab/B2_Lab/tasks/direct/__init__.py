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
