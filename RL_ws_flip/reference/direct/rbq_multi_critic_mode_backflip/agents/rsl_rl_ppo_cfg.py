# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoActorCriticRecurrentCfg, RslRlPpoAlgorithmCfg

from dataclasses import MISSING

# @configclass
# class RslRLPpoActorCriticCustomCfg(RslRlPpoActorCriticRecurrentCfg):
#     class_name: str = "ActorCriticCustom"
#     """The policy class name. Default is ActorCriticCustom."""

#     rollout_class_name: str = "RolloutStorageCustom"

#     rnn_type: str = MISSING
#     """The type of RNN to use. Either "lstm" or "gru"."""

#     rnn_hidden_dim: int = MISSING
#     """The dimension of the RNN layers."""

#     rnn_num_layers: int = MISSING
#     """The number of RNN layers."""

#     num_history_encode_dim: int = 16

#     history_encoder_hidden_dims: list[int]  = [256, 256, 256]

@configclass
class RslRLPpoActorMultiCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticMultiCritic"
    """The policy class name. Default is ActorCriticCustom."""

    rollout_class_name: str = "RolloutStorageMultiCritic"

    list_of_actors=["actor"],
    list_of_critics : list[str] = ["critic"]

@configclass
class RBQPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 6
    max_iterations = 500000
    save_interval = 50
    experiment_name = "rbq_vel_multi_flip_direct"

    obs_groups = {
            "policy": ["policy"],
            "stand_up_critic": ["critic"],
            "recovery_critic": ["critic"],
            "quadruped_critic": ["critic"],
            "biped_critic": ["critic"],
            "safety_critic": ["critic"]
        }
    
    policy = RslRLPpoActorMultiCriticCfg(
        class_name = "ActorCriticMultiCritic",
        rollout_class_name = "RolloutStorageMultiCritic",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        # actor_hidden_dims=[1024, 512, 256],
        # critic_hidden_dims=[1024, 512, 256],
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        list_of_critics=["stand_up_critic", 
                         "recovery_critic", 
                         "quadruped_critic", 
                         "biped_critic",
                         "safety_critic"]
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPOMultiCritic",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef= 0.005, #0.005,
        num_learning_epochs=3,
        num_mini_batches=20,
        learning_rate=1.0e-3, #1.0e-3
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# @configclass
# class RBQPPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 24
#     max_iterations = 500000
#     save_interval = 50
#     experiment_name = "rbq_vel_direct"
#     obs_groups = {"policy": ["policy"], 
#                  "critic": ["critic"],
#                  "history_encoder_input": ["history_encoder_input"],
#                  }
    
#     policy = RslRLPpoActorCriticCustomCfg(
#         class_name="ActorCriticCustom",
#         init_noise_std=1.0,
#         actor_obs_normalization=False,
#         critic_obs_normalization=False,
#         actor_hidden_dims=[512, 256, 128],
#         critic_hidden_dims=[512, 256, 128],
#         activation="elu",
#         rnn_type = "gru",
#         rnn_hidden_dim = 64,
#         rnn_num_layers = 1,
#         num_history_encode_dim = 16,
#         history_encoder_hidden_dims = [128, 64],

#     )

#     algorithm = RslRlPpoAlgorithmCfg(
#         class_name="PPOCustom",
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.005,
#         num_learning_epochs=5,
#         num_mini_batches=4,
#         learning_rate=1.0e-3,
#         schedule="adaptive",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#     )