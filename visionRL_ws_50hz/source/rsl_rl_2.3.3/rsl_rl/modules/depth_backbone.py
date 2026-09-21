# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Extreme Parkour depth backbone, ported/adapted for rsl_rl_2.3.3 + IsaacLab B2.
# Reference: Cheng et al., "Extreme Parkour with Legged Robots" (ICRA 2024),
#            https://github.com/chengxuxin/extreme-parkour
#
# Phase 2 (student/vision): a CNN encodes the forward depth image into a latent
# that is trained to match the teacher's scandots latent, plus a small yaw head.
# A GRU carries temporal context over the low-frequency, jittery depth stream.

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


class DepthOnlyFCBackbone(nn.Module):
    """CNN that maps a single depth frame [B, 1, H, W] -> [B, output_dim].

    Default geometry matches Extreme Parkour's 58x87 resized depth image, but the
    flatten dimension is computed dynamically so any (H, W) works.
    """

    def __init__(self, output_dim: int, height: int = 58, width: int = 87, activation: str = "elu"):
        super().__init__()
        act = resolve_nn_activation(activation)
        self.output_dim = output_dim

        self.image_compression = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            act,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            act,
            nn.Flatten(),
        )

        # Compute the flattened conv output size with a dummy forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, height, width)
            flat_dim = self.image_compression(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(flat_dim, 128),
            act,
            nn.Linear(128, output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, H, W] or [B, 1, H, W]
        if images.dim() == 3:
            images = images.unsqueeze(1)
        x = self.image_compression(images)
        return self.linear(x)


class RecurrentDepthBackbone(nn.Module):
    """Depth encoder with proprioception fusion + GRU memory.

    forward(depth_image, proprio) -> latent of size (scandots_latent_dim + 2),
    where the last two entries are the predicted (delta_yaw, delta_next_yaw).
    The hidden state is kept internally and cleared per-env on `reset(dones)`.
    """

    def __init__(
        self,
        num_prop: int,
        scandots_latent_dim: int = 32,
        depth_height: int = 58,
        depth_width: int = 87,
        rnn_hidden_dim: int = 512,
        activation: str = "elu",
    ):
        super().__init__()
        act = resolve_nn_activation(activation)
        self.scandots_latent_dim = scandots_latent_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        self.base_backbone = DepthOnlyFCBackbone(
            output_dim=scandots_latent_dim, height=depth_height, width=depth_width, activation=activation
        )
        # Fuse depth latent with proprioception before the recurrent layer.
        self.combination = nn.Sequential(
            nn.Linear(scandots_latent_dim + num_prop, 128),
            act,
            nn.Linear(128, scandots_latent_dim),
        )
        self.rnn = nn.GRU(input_size=scandots_latent_dim, hidden_size=rnn_hidden_dim, batch_first=True)
        self.output_mlp = nn.Sequential(
            nn.Linear(rnn_hidden_dim, scandots_latent_dim + 2),
            nn.Tanh(),
        )
        self.hidden_states = None

    def detach_hidden_states(self):
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.detach()

    def reset(self, dones=None):
        if self.hidden_states is None or dones is None:
            return
        # dones: bool/byte tensor [num_envs]; zero-out hidden state of finished envs.
        self.hidden_states[..., dones.bool(), :] = 0.0

    def forward(self, depth_image: torch.Tensor, proprioception: torch.Tensor) -> torch.Tensor:
        depth_latent = self.base_backbone(depth_image)                      # [B, scandots_latent_dim]
        fused = self.combination(torch.cat([depth_latent, proprioception], dim=-1))  # [B, scandots_latent_dim]
        # GRU expects [B, T, F]; we run one timestep at a time and keep the hidden state.
        out, self.hidden_states = self.rnn(fused.unsqueeze(1), self.hidden_states)
        latent = self.output_mlp(out.squeeze(1))                            # [B, scandots_latent_dim + 2]
        return latent
