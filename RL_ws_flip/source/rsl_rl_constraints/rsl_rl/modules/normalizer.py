# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#  Copyright (c) 2020 Preferred Networks, Inc.

from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):

    def __init__(self, shape, eps=1e-2, until=None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        if self.training:
            self.update(x)
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""

        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count

        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean


class EmpiricalHistoryNormalization(nn.Module):
    def __init__(self, obs_shape, history_length, eps=1e-2, until=None):
        super().__init__()
        self.obs_shape = obs_shape if isinstance(obs_shape, (list, tuple)) else (obs_shape,)
        self.history_length = history_length
        # Normalizer for single observation
        self.obs_normalizer = EmpiricalNormalization(obs_shape, eps, until)
        self.register_buffer("history_buffer", None, persistent=False)

    def forward(self, obs):
        batch_size = obs.shape[0]

        # Initialize history buffer if needed
        if self.history_buffer is None:
            # obs_shape를 정수로 변환
            if isinstance(self.obs_shape, (list, tuple)):
                total_obs_dim = 1
                for dim in self.obs_shape:
                    total_obs_dim *= int(dim)  # 각 차원을 int로 변환
            else:
                total_obs_dim = int(self.obs_shape)

            # 모든 차원을 int로 변환
            self.history_buffer = torch.zeros(
                int(batch_size), int(self.history_length), int(total_obs_dim),
                device=obs.device, dtype=obs.dtype
            )

        # Normalize current observation
        normalized_obs = self.obs_normalizer(obs)

        # Reshape for history buffer
        normalized_obs_flat = normalized_obs.view(batch_size, -1)

        # Shift history and add new observation
        self.history_buffer[:, :-1] = self.history_buffer[:, 1:].clone()
        self.history_buffer[:, -1] = normalized_obs_flat
        # Return flattened history (batch_size, history_length * obs_dim)
        return self.history_buffer.view(batch_size, -1)

    def reset_history(self, env_ids=None):
        if self.history_buffer is not None:
            if env_ids is None:
                self.history_buffer.zero_()
            else:
                self.history_buffer[env_ids] = 0.0

    @property
    def mean(self):
        return self.obs_normalizer.mean

    @property 
    def std(self):
        return self.obs_normalizer.std


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Reward normalization from Pathak's large scale study on PPO.

    Reward normalization. Since the reward function is non-stationary, it is useful to normalize
    the scale of the rewards so that the value function can learn quickly. We did this by dividing
    the rewards by a running estimate of the standard deviation of the sum of discounted rewards.
    """

    def __init__(self, shape, eps=1e-2, gamma=0.99, until=None):
        super().__init__()

        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        self.disc_avg = DiscountedAverage(gamma)

    def forward(self, rew):
        if self.training:
            # update discounected rewards
            avg = self.disc_avg.update(rew)

            # update moments from discounted rewards
            self.emp_norm.update(avg)

        if self.emp_norm._std > 0:
            return rew / self.emp_norm._std
        else:
            return rew


class DiscountedAverage:
    r"""Discounted average of rewards.

    The discounted average is defined as:

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t

    Args:
        gamma (float): Discount factor.
    """

    def __init__(self, gamma):
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        return self.avg
