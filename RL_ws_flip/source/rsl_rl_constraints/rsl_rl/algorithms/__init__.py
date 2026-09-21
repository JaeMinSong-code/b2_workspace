# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

# from .ppo_cnet import PPO
from .ppo_historyencoder import PPO
# from .ppo_cnet import PPO

__all__ = ["PPO"]
