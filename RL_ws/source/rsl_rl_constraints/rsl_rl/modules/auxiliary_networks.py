from __future__ import annotations

import torch
import torch.nn as nn


def build_mlp(in_dim, hidden_dims, out_dim=None, activation=nn.ELU()):
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(activation)
        prev = h
    if out_dim is not None:
        layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class AuxiliaryNetworks(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_velocity_estimator_obs,
        num_priv_encoder_obs,
        num_height_encoder_obs,
        num_cenet_obs,
        num_cenet_obs_history,
        cenet_z_dim,
        velocity_estimator_hidden_dims=[128, 128],
        privileged_encoder_hidden_dims=[128, 64],
        height_encoder_hidden_dims=[80, 60],
        cenet_encoder_hidden_dims=(128, 64),
        cenet_decoder_hidden_dims=(64, 128),
        **kwargs,
    ):
        super().__init__()

        activation = nn.ELU()
        init_noise_std=1.0,

        self.velocity_estimator = build_mlp(
            num_velocity_estimator_obs,
            velocity_estimator_hidden_dims,
            kwargs["velocity_estimator_output_dim"],
            activation,
        )

        self.privileged_encoder = build_mlp(
            num_priv_encoder_obs,
            privileged_encoder_hidden_dims,
            kwargs["priv_encoder_output_dim"],
            activation,
        )

        self.height_encoder = build_mlp(
            num_height_encoder_obs,
            height_encoder_hidden_dims,
            kwargs["height_encoder_output_dim"],
            activation,
        )

        self.cenet_v_dim = kwargs["velocity_estimator_output_dim"]
        self.cenet_z_dim = cenet_z_dim
        self.cenet_code_dim = self.cenet_v_dim + self.cenet_z_dim
        self.cenet_in_dim = num_cenet_obs_history

        self.cenet_encoder = build_mlp(
            self.cenet_in_dim,
            cenet_encoder_hidden_dims,
            out_dim=None,
            activation=activation,
        )
        enc_out_dim = cenet_encoder_hidden_dims[-1]

        self.cenet_mean_vel = nn.Linear(enc_out_dim, self.cenet_v_dim)
        # self.cenet_logvar_vel = nn.Linear(enc_out_dim, self.cenet_v_dim)

        self.cenet_mean_latent = nn.Linear(enc_out_dim, self.cenet_z_dim)
        self.cenet_logvar_latent = nn.Linear(enc_out_dim, self.cenet_z_dim)

        self.cenet_decoder = build_mlp(
            self.cenet_code_dim,
            cenet_decoder_hidden_dims,
            out_dim=num_cenet_obs,
            activation=activation,
        )

        # print(f"velocity_estimator : {self.velocity_estimator}")
        # print(f"privileged_encoder : {self.privileged_encoder}")
        # print(f"height_encoder : {self.height_encoder}")
        # print(f"cenet_encoder : {self.cenet_encoder}")

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @torch.jit.export
    def infer_priv_latent(self, obs_priv):
        priv = obs_priv
        return self.privileged_encoder(priv)

    @torch.jit.export
    def infer_height_latent(self, obs_heights):
        heights = obs_heights
        return self.height_encoder(heights)

    @torch.jit.export
    def infer_velocity(self, obs_velocity_estimator):
        obs = obs_velocity_estimator
        return self.velocity_estimator(obs)

    @staticmethod
    def _reparameterise(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def cenet_forward_train(self, obs_history):
        h = self.cenet_encoder(obs_history)

        mean_vel = self.cenet_mean_vel(h)
        # logvar_vel = self.cenet_logvar_vel(h)

        mean_latent = self.cenet_mean_latent(h)
        logvar_latent = self.cenet_logvar_latent(h)
        # vel = self._reparameterise(mean_vel, logvar_vel) 
        vel = mean_vel
        latent = self._reparameterise(mean_latent, logvar_latent)

        code = torch.cat([vel, latent], dim=-1)
        recon = self.cenet_decoder(code)

        return {
            "code": code,
            "vel": vel,
            "latent": latent,
            "mean_vel": mean_vel,
            # "logvar_vel": logvar_vel,
            "mean_latent": mean_latent,
            "logvar_latent": logvar_latent,
            "recon": recon,
        }

    @torch.jit.export
    def cenet_infer(self, obs_history):
        h = self.cenet_encoder(obs_history)
        mean_vel = self.cenet_mean_vel(h)
        mean_latent = self.cenet_mean_latent(h)
        return mean_vel, mean_latent
