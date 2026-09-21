from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage


class PPO:

    def __init__(
        self,
        actor,
        reward_critic,
        cost_critic,
        aux_networks,
        num_cost,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        velocity_loss_coef=1.0,
        priv_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        gae_coeff=0.97,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # # Symmetry parameters
        # symmetry_cfg: dict | None = None,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.gae_coeff = gae_coeff
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.velocity_loss_coef = velocity_loss_coef
        self.priv_loss_coef = priv_loss_coef
        self.num_cost = num_cost

        # Constraint parameters (like agent.py)
        self.con_coeff = 10.0  # Make this configurable
        # Multi-step penalty parameters for early termination
        self.multistep_penalty_steps = 5  # 조기 종료 전 N스텝에 페널티 적용
        self.multistep_penalty_decay = 0.7  # 과거 스텝으로 갈수록 페널티 감소
        self.early_termination_penalty = 3.0  # 기본 페널티 배수

        # Different thresholds for different cost types
        base_thresholds = torch.ones(num_cost) * 0.01  # Default threshold
        base_thresholds[1] = 0.01
        # Active cost order: c1com_height, c2dof_pos, c3dof_vel,
        # c5gait_pattern, c6undesired_contact
        base_thresholds[3] = 0.15

        self.con_thresholds = (base_thresholds / (1.0 - gamma)).to(device)  # Pre-normalize

        # PPO components
        self.actor = actor
        self.actor.to(device)
        self.reward_critic = reward_critic
        self.reward_critic.to(device)
        self.cost_critic = cost_critic
        self.cost_critic.to(device)

        # Create optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.reward_critic_optimizer = optim.Adam(
            self.reward_critic.parameters(), lr=learning_rate
        )
        self.cost_critic_optimizer = optim.Adam(
            self.cost_critic.parameters(), lr=learning_rate
        )
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # aux networks
        self.aux_networks = aux_networks
        self.aux_networks.to(self.device)
        self.aux_networks_optimizer = optim.Adam(
            self.aux_networks.parameters(), lr=1e-4
        )

    def init_storage(
        self,
        num_envs,
        num_cost,
        num_transitions_per_env,
        prop_obs_shape,
        prop_obs_history_shape,
        actor_obs_shape,
        critic_obs_shape,
        height_obs_shape,
        velocity_estimator_obs_shape,
        velocity_estimator_target_shape,
        priv_obs_shape,
        priv_obs_target_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_cost,
            num_transitions_per_env,
            prop_obs_shape,
            prop_obs_history_shape,
            actor_obs_shape,
            critic_obs_shape,
            height_obs_shape,
            velocity_estimator_obs_shape,
            velocity_estimator_target_shape,
            priv_obs_shape,
            priv_obs_target_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor.test()

    def train_mode(self):
        self.actor.train()

    def act(
        self,
        porprio_obs,
        porprio_obs_history,
        critic_obs,
        velocity_estimator_obs,
        velocity_estimator_target,
        priv_obs,
        priv_obs_target,
        heights_obs,
    ):
        with torch.no_grad():
            v_hat, z_hat = self.aux_networks.cenet_infer(porprio_obs_history)
        # v_est = self.aux_networks.infer_velocity(velocity_estimator_obs)
        # priv_hat = self.aux_networks.infer_priv_latent(priv_obs)

        actor_obs = torch.cat([porprio_obs, v_hat, z_hat], dim=-1,)
        # Compute the actions and values
        self.transition.actions = self.actor.act(actor_obs).detach()
        self.transition.reward_values = self.reward_critic.evaluate(critic_obs).detach()
        self.transition.cost_values = self.cost_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor.action_mean.detach()
        self.transition.action_sigma = self.actor.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.prop_observations = porprio_obs
        self.transition.prop_observation_history = porprio_obs_history
        self.transition.actor_observations = actor_obs
        self.transition.critic_observations = critic_obs
        self.transition.velocity_estimator_obs = velocity_estimator_obs
        self.transition.velocity_estimator_target = velocity_estimator_target
        self.transition.priv_obs = priv_obs
        self.transition.priv_obs_target = priv_obs_target
        self.transition.height_obs = heights_obs

        return self.transition.actions

    def process_env_step(self, rewards, costs, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.costs = costs.clone()

        self.transition.dones = dones
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.reward_values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )
            self.transition.costs += self.gamma * torch.squeeze(
                self.transition.cost_values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_reward_values = self.reward_critic.evaluate(last_critic_obs).detach()
        last_cost_values = self.cost_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_reward_values,
            last_cost_values,
            self.gamma,
            self.lam,
            normalize_advantage=False,
        )

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_cost_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_velocity_estimator_loss = 0
        mean_autoenc_loss = 0
        mean_vel_loss = 0
        mean_recon_loss = 0
        beta = 0.01
        
        rollout_cost_returns = self.storage.cost_targets.flatten(0, 1)
        rollout_cost_advantages = self.storage.cost_advantages.flatten(0, 1)
        con_vals = rollout_cost_returns.mean(dim=0)
        # rollout_costs = self.storage.costs.flatten(0, 1)
        # raw_cost_vals = rollout_costs.mean(dim=0)
        rollout_cost_adv_mean = rollout_cost_advantages.mean(dim=0, keepdim=True)
        rollout_cost_adv_std = rollout_cost_advantages.std(dim=0, keepdim=True)

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        # iterate over batches
        for (
            prop_obs_batch,
            next_prop_obs_batch,
            prop_obs_history_batch,
            actor_obs_batch,
            critic_obs_batch,
            velocity_estimator_obs_batch,
            velocity_estimator_target_batch,
            priv_obs_batch,
            priv_obs_target_batch,
            height_obs_batch,
            actions_batch,
            reward_target_values_batch,
            cost_target_values_batch,
            reward_advantages_batch,
            cost_advantages_batch,
            reward_returns_batch,
            cost_returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            dones_batch,
        ) in generator:

            original_batch_size = actor_obs_batch.shape[0]
            # -- actor
            self.actor.act(actor_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
            # -- critic
            reward_value_batch = self.reward_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            cost_value_batch = self.cost_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )

            # -- auxiliary_networks
            # # 1) Concurrent estimator networks
            # estimated_velocity_batch = self.aux_networks.infer_velocity(
            #     velocity_estimator_obs_batch
            # )
            # privileged_encoder_batch = self.aux_networks.infer_priv_latent(
            #     priv_obs_batch
            # )
            # 2) VAE_networks
            cenet_out = self.aux_networks.cenet_forward_train(prop_obs_history_batch)            # Beta VAE loss

            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.actor.action_mean[:original_batch_size]
            sigma_batch = self.actor.action_std[:original_batch_size]
            entropy_batch = self.actor.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.actor_optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
                    for param_group in self.reward_critic_optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
                    for param_group in self.cost_critic_optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Value function loss
            if self.use_clipped_value_loss:
                reward_value_clipped = reward_target_values_batch + (
                    reward_value_batch - reward_target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                reward_value_losses = (reward_value_batch - reward_returns_batch).pow(2)
                reward_value_losses_clipped = (
                    reward_value_clipped - reward_returns_batch
                ).pow(2)
                reward_value_loss = torch.max(
                    reward_value_losses, reward_value_losses_clipped
                ).mean()
            else:
                reward_value_loss = (
                    (reward_value_batch - reward_returns_batch).pow(2).mean()
                )
            reward_value_loss = self.value_loss_coef * reward_value_loss
            cost_value_loss = torch.nn.functional.smooth_l1_loss(
                cost_returns_batch, cost_value_batch
            )

            # Reward Critic Gradient step
            self.reward_critic_optimizer.zero_grad()
            reward_value_loss.backward()
            nn.utils.clip_grad_norm_(
                self.reward_critic.parameters(), self.max_grad_norm
            )
            self.reward_critic_optimizer.step()

            # Cost Critic Gradient step
            self.cost_critic_optimizer.zero_grad()
            cost_value_loss.backward()
            nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
            self.cost_critic_optimizer.step()

            ####################################################################################################
            # auxiliary_networks loss
            ####################################################################################################
            # # 1) Concurrent estimator loss
            # velocity_estimator_loss = torch.nn.functional.smooth_l1_loss(
            #     velocity_estimator_target_batch.detach(), estimated_velocity_batch
            # ) + torch.nn.functional.smooth_l1_loss(
            #     priv_obs_target_batch.detach(), privileged_encoder_batch
            # )
            # 2) CENet loss
            vel_loss = F.smooth_l1_loss(cenet_out["mean_vel"], velocity_estimator_target_batch.detach())
            # done(에피소드 경계) 전이는 next_prop_obs가 리셋 후 obs라 recon 대상에서 제외
            recon_valid = 1.0 - dones_batch.float()                                              # [B,1]
            recon_per_elem = F.mse_loss(cenet_out["recon"], next_prop_obs_batch.detach(), reduction="none")
            recon_loss = (recon_per_elem * recon_valid).sum() / (recon_valid.sum().clamp(min=1.0) * recon_per_elem.shape[-1])
            kl_latent = -0.5 * torch.mean(1.0 + cenet_out["logvar_latent"] - cenet_out["mean_latent"].pow(2) - cenet_out["logvar_latent"].exp())
            autoenc_loss = vel_loss + recon_loss + beta * kl_latent
            #################################################################################################### 

            # Surrogate loss
            reward_advantages_batch -= reward_advantages_batch.mean(dim=0, keepdim=True)
            reduced_gaes_tensor = (reward_advantages_batch).sum(dim=-1)
            reduced_gaes_tensor /= reduced_gaes_tensor.std() + 1e-8
            cost_gaes_tensor = (
                cost_advantages_batch - rollout_cost_adv_mean
            ) / (rollout_cost_adv_std + 1e-8)

            # Apply constraint penalty like agent.py
            for cost_idx in range(self.num_cost):
                if con_vals[cost_idx] > self.con_thresholds[cost_idx]:
                    reduced_gaes_tensor -= self.con_coeff * cost_gaes_tensor[:, cost_idx]
            reduced_gaes_tensor /= reduced_gaes_tensor.std() + 1e-8

            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(reduced_gaes_tensor) * ratio
            surrogate_clipped = -torch.squeeze(reduced_gaes_tensor) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            actor_loss = surrogate_loss - (self.entropy_coef * entropy_batch.mean())

            # Actor networks Gradient step
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # auxiliary_networks Gradient step
            # # 1) Concurrent estimator Gradient step
            # self.aux_networks_optimizer.zero_grad()
            # velocity_estimator_loss.backward()
            # nn.utils.clip_grad_norm_(self.aux_networks.parameters(), self.max_grad_norm)
            # self.aux_networks_optimizer.step()
            # 1) VAW Gradient step
            self.aux_networks_optimizer.zero_grad()
            autoenc_loss.backward()
            self.aux_networks_optimizer.step()

            # Store the losses
            mean_value_loss += reward_value_loss.item()
            mean_cost_value_loss += cost_value_loss.mean(dim=-1).item()
            mean_entropy += entropy_batch.mean().item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_autoenc_loss += autoenc_loss.item()
            mean_vel_loss += vel_loss.item()
            mean_recon_loss += recon_loss.item()
            # mean_velocity_estimator_loss += velocity_estimator_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_cost_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # mean_velocity_estimator_loss /= num_updates
        mean_autoenc_loss /= num_updates
        mean_vel_loss /= num_updates
        mean_recon_loss /= num_updates
        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "mean_cost_value_loss": mean_cost_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "autoencoder": mean_autoenc_loss,
            # "mean_velocity_estimator_loss": mean_velocity_estimator_loss,
            "mean_vel_loss": mean_vel_loss,
            "mean_recon_loss": mean_recon_loss,
        }

        return loss_dict, con_vals.detach().cpu()
