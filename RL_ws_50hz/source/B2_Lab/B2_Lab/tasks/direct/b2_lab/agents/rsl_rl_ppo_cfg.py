from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class B2LabFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):

    experiment_name: str = "b2"
    num_steps_per_env: int = 24   # 50Hz policy: 24 step = 0.48s (기존 48@100Hz와 동일 시간 horizon)
    max_iterations: int = 100000
    save_interval: int = 50
    empirical_normalization: bool = False
    num_cost: int = 5
    clip_actions: float = 10.0
    history_length: int = 10.0

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    aux_network = dict(
        class_name='AuxiliaryNetworks',
        velocity_estimator_hidden_dims=[128, 128],
        # privileged_encoder_hidden_dims=[64, 32],
        privileged_encoder_hidden_dims=[128, 64],
        height_encoder_hidden_dims=[80, 60],
        velocity_estimator_output_dim=3,
        priv_encoder_output_dim=32,
        height_encoder_output_dim=96,
        activation='elu'
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # entropy_coef=0.01,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
