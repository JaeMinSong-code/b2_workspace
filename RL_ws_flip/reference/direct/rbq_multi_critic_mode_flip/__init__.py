# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RBQ-MULTI-MODE-FLIP",
    entry_point=f"{__name__}.rbq_multi_critic_env:RBQEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rbq_multi_critic_env_cfg:RBQEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RBQPPORunnerCfg",
    },
)


# gym.register(
#     id="RBQ-SAMPLING-PLAY",
#     entry_point=f"{__name__}.rbq_sampling_env:RBQEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.rbq_sampling_env_cfg:RBQEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RBQPPORunnerCfg",
#     },
# )

