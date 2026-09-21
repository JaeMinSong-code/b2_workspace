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


# =============================================================================
# Extreme Parkour teacher (phase 1) runner config — uses rsl_rl_2.3.3 with the
# custom ActorCriticParkour policy. Train with:
#   train_b2.py --task B2-Parkour --parkour --rsl_rl_type original
# =============================================================================


@configclass
class ParkourActorCriticCfg:
    """Policy cfg consumed by rsl_rl_2.3.3 OnPolicyRunner -> ActorCriticParkour."""

    class_name: str = "ActorCriticParkour"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    actor_hidden_dims: list = [512, 256, 128]
    critic_hidden_dims: list = [512, 256, 128]
    activation: str = "elu"
    # observation layout (must match B2LabParkourEnvCfg / env _get_observations)
    num_prop: int = 49
    num_scan: int = 121
    num_priv: int = 3
    num_priv_latent: int = 33
    num_hist: int = 10
    # encoder dims
    scan_encoder_dims: list = [128, 64, 32]
    priv_encoder_dims: list = [64, 20]


@configclass
class B2LabParkourPPORunnerCfg(RslRlOnPolicyRunnerCfg):

    experiment_name: str = "b2_parkour"
    num_steps_per_env: int = 24
    max_iterations: int = 100000
    save_interval: int = 100
    empirical_normalization: bool = False
    clip_actions: float = 10.0

    policy = ParkourActorCriticCfg()

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
