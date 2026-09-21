# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Extreme-Parkour style ActorCritic for rsl_rl_2.3.3.
# Reference: Cheng et al., "Extreme Parkour with Legged Robots" (ICRA 2024).
#
# The policy observation is a single flat vector laid out as:
#   [ proprio (num_prop) | scandots (num_scan) | priv_explicit (num_priv)
#     | priv_latent (num_priv_latent) | history (num_hist * num_prop) ]
#
# The actor internally:
#   - encodes scandots      -> scan_latent   (via scan_encoder)
#   - encodes priv_latent   -> priv_latent_e (via priv_encoder)   [phase 1 / teacher]
#     OR estimates it from proprio history    (via history_encoder) [RMA / adaptation]
#   - concatenates [proprio, scan_latent, priv_explicit, priv_latent_e]
#   - runs the actor MLP -> action mean
#
# This class is a drop-in `policy` for rsl_rl_2.3.3's PPO: it exposes
# act / act_inference / evaluate / get_actions_log_prob / action_mean /
# action_std / entropy / reset, and is_recurrent = False.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


def _build_mlp(input_dim, hidden_dims, output_dim, activation, output_activation=None):
    layers = [nn.Linear(input_dim, hidden_dims[0]), activation]
    for i in range(len(hidden_dims)):
        if i == len(hidden_dims) - 1:
            layers.append(nn.Linear(hidden_dims[i], output_dim))
        else:
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(activation)
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class StateHistoryEncoder(nn.Module):
    """1D temporal conv over the proprioception history (RMA adaptation module)."""

    def __init__(self, num_prop: int, num_hist: int, output_dim: int, activation: str = "elu", channel_size: int = 10):
        super().__init__()
        act = resolve_nn_activation(activation)
        self.num_prop = num_prop
        self.num_hist = num_hist

        self.tsteps_encoder = nn.Sequential(nn.Linear(num_prop, 3 * channel_size), act)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=4, stride=2),
            act,
            nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=2, stride=1),
            act,
            nn.Flatten(),
        )
        # Dynamically size the final linear layer.
        with torch.no_grad():
            dummy = torch.zeros(1, num_hist, num_prop)
            proj = self.tsteps_encoder(dummy)              # [1, T, 3C]
            proj = proj.permute(0, 2, 1)                   # [1, 3C, T]
            flat = self.conv(proj).shape[1]
        self.linear_output = nn.Sequential(nn.Linear(flat, output_dim), act)

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        # obs_history: [B, num_hist, num_prop]
        proj = self.tsteps_encoder(obs_history)            # [B, T, 3C]
        proj = proj.permute(0, 2, 1)                       # [B, 3C, T]
        feat = self.conv(proj)                             # [B, C*L]
        return self.linear_output(feat)                    # [B, output_dim]


class ActorCriticParkour(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        # --- observation layout (must sum to num_actor_obs) ---
        num_prop=49,
        num_scan=100,
        num_priv=3,
        num_priv_latent=33,
        num_hist=10,
        # --- encoder / backbone dims ---
        scan_encoder_dims=[128, 64, 32],
        priv_encoder_dims=[64, 20],
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print("ActorCriticParkour got unexpected arguments, ignored: " + str(list(kwargs.keys())))
        super().__init__()
        act = resolve_nn_activation(activation)

        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_priv = num_priv
        self.num_priv_latent = num_priv_latent
        self.num_hist = num_hist
        self.scan_latent_dim = scan_encoder_dims[-1]
        self.priv_latent_out = priv_encoder_dims[-1]

        expected = num_prop + num_scan + num_priv + num_priv_latent + num_hist * num_prop
        assert num_actor_obs == expected, (
            f"num_actor_obs ({num_actor_obs}) != layout sum ({expected}). "
            f"prop={num_prop}, scan={num_scan}, priv={num_priv}, priv_latent={num_priv_latent}, "
            f"hist={num_hist}*{num_prop}"
        )

        # Scandots (heightmap) encoder -> scan latent.
        self.scan_encoder = _build_mlp(
            num_scan, scan_encoder_dims[:-1], scan_encoder_dims[-1], act, output_activation=nn.Tanh()
        )
        # Privileged-latent encoder (friction/mass/etc.) -> priv latent (teacher path).
        self.priv_encoder = _build_mlp(num_priv_latent, priv_encoder_dims[:-1], priv_encoder_dims[-1], act)
        # RMA adaptation: estimate priv latent from proprio history (student/deploy path).
        self.history_encoder = StateHistoryEncoder(num_prop, num_hist, self.priv_latent_out, activation)

        # Actor backbone input = proprio + scan_latent + priv_explicit + priv_latent
        actor_in = num_prop + self.scan_latent_dim + num_priv + self.priv_latent_out
        self.actor = _build_mlp(actor_in, actor_hidden_dims, num_actions, act)

        # Critic sees the full privileged observation vector directly.
        self.critic = _build_mlp(num_critic_obs, critic_hidden_dims, 1, act)

        # Action noise.
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}")

        self.distribution = None
        Normal.set_default_validate_args(False)

        print(f"[ActorCriticParkour] actor_in={actor_in}, scan_latent={self.scan_latent_dim}, "
              f"priv_latent={self.priv_latent_out}, critic_in={num_critic_obs}")

    # ------------------------------------------------------------------ helpers
    def _split(self, obs):
        p = self.num_prop
        s = self.num_scan
        pe = self.num_priv
        pl = self.num_priv_latent
        prop = obs[:, :p]
        scan = obs[:, p:p + s]
        priv_explicit = obs[:, p + s:p + s + pe]
        priv_latent = obs[:, p + s + pe:p + s + pe + pl]
        history = obs[:, -self.num_hist * p:].view(-1, self.num_hist, p)
        return prop, scan, priv_explicit, priv_latent, history

    def infer_scandots_latent(self, obs):
        _, scan, _, _, _ = self._split(obs)
        return self.scan_encoder(scan)

    def infer_priv_latent(self, obs):
        _, _, _, priv_latent, _ = self._split(obs)
        return self.priv_encoder(priv_latent)

    def infer_hist_latent(self, obs):
        _, _, _, _, history = self._split(obs)
        return self.history_encoder(history)

    def adaptation_loss(self, obs):
        """RMA / regularized online adaptation.

        Trains the proprio-history encoder to regress the (detached) privileged
        latent produced by the privileged encoder. After phase-1 training the
        history encoder can stand in for the privileged encoder at deployment
        (student runs with hist_encoding=True, no privileged info needed).
        Gradient flows only to the history encoder (target is detached).
        """
        _, _, _, priv_latent, history = self._split(obs)
        with torch.no_grad():
            target = self.priv_encoder(priv_latent)
        pred = self.history_encoder(history)
        return F.mse_loss(pred, target)

    def _actor_forward(self, obs, hist_encoding: bool = False, scan_latent=None):
        prop, scan, priv_explicit, priv_latent, history = self._split(obs)
        if scan_latent is None:
            scan_latent = self.scan_encoder(scan)
        if hist_encoding:
            priv_latent_e = self.history_encoder(history)
        else:
            priv_latent_e = self.priv_encoder(priv_latent)
        actor_in = torch.cat([prop, scan_latent, priv_explicit, priv_latent_e], dim=-1)
        return self.actor(actor_in)

    # ------------------------------------------------------------------ rsl_rl API
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, hist_encoding: bool = False):
        mean = self._actor_forward(observations, hist_encoding=hist_encoding)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations, hist_encoding: bool = False, **kwargs):
        self.update_distribution(observations, hist_encoding=hist_encoding)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, hist_encoding: bool = False, scan_latent=None):
        return self._actor_forward(observations, hist_encoding=hist_encoding, scan_latent=scan_latent)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
