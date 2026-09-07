# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.prop_observations = None
            self.next_prop_observations = None
            self.prop_observation_history = None
            self.actor_observations = None
            self.critic_observations = None
            self.velocity_estimator_obs = None
            self.velocity_estimator_target = None
            self.priv_obs = None
            self.priv_obs_target = None
            self.height_obs = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.reward_values = None
            self.cost_values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.costs = None

        def clear(self):
            self.__init__()

    def __init__(
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
        actions_shape,
        device="cpu",
    ):
        # store inputs
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.prop_obs_shape = prop_obs_shape
        self.prop_obs_history_shape = prop_obs_history_shape
        self.actor_obs_shape = actor_obs_shape
        self.critic_obs_shape = critic_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.prop_obs = torch.zeros(num_transitions_per_env, num_envs, *prop_obs_shape, device=self.device)
        self.prop_obs_history = torch.zeros(num_transitions_per_env, num_envs, *prop_obs_history_shape, device=self.device)
        self.actor_obs = torch.zeros(num_transitions_per_env, num_envs, *actor_obs_shape, device=self.device)
        self.critic_obs = torch.zeros(num_transitions_per_env, num_envs, *critic_obs_shape, device=self.device)

        self.height_obs = torch.zeros(num_transitions_per_env, num_envs, *height_obs_shape, device=self.device)
        self.velocity_estimator_obs = torch.zeros(num_transitions_per_env, num_envs, *velocity_estimator_obs_shape, device=self.device)
        self.velocity_estimator_target = torch.zeros(num_transitions_per_env, num_envs, *velocity_estimator_target_shape, device=self.device)
        self.priv_obs = torch.zeros(num_transitions_per_env, num_envs, *priv_obs_shape, device=self.device)
        self.priv_obs_target = torch.zeros(num_transitions_per_env, num_envs, *priv_obs_target_shape, device=self.device)
        self.next_prop_obs = torch.zeros(num_transitions_per_env, num_envs, *prop_obs_shape, device=self.device)

        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.reward_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.reward_targets = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.reward_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # for N-P30
        self.costs = torch.zeros(num_transitions_per_env, num_envs, num_cost, device=self.device)
        self.cost_targets = torch.zeros(num_transitions_per_env, num_envs, num_cost, device=self.device)
        self.cost_values = torch.zeros(num_transitions_per_env, num_envs, num_cost, device=self.device)
        self.cost_advantages = torch.zeros(num_transitions_per_env, num_envs, num_cost, device=self.device)

        self.step = 0

    def add_transitions(self, transition: Transition):
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")
        # Core
        self.prop_obs[self.step].copy_(transition.prop_observations)
        self.prop_obs_history[self.step].copy_(transition.prop_observation_history)
        self.actor_obs[self.step].copy_(transition.actor_observations)
        self.critic_obs[self.step].copy_(transition.critic_observations)

        self.velocity_estimator_obs[self.step].copy_(transition.velocity_estimator_obs)
        self.velocity_estimator_target[self.step].copy_(transition.velocity_estimator_target)
        self.priv_obs[self.step].copy_(transition.priv_obs)
        self.priv_obs_target[self.step].copy_(transition.priv_obs_target)
        self.height_obs[self.step].copy_(transition.height_obs)
        self.next_prop_obs[self.step].copy_(transition.next_prop_observations)

        # For PPO
        self.actions[self.step].copy_(transition.actions)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.reward_values[self.step].copy_(transition.reward_values)

        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)

        # For P30
        self.costs[self.step].copy_(transition.costs)
        self.cost_values[self.step].copy_(transition.cost_values)

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_reward_values, last_cost_values, gamma, lam, normalize_advantage: bool = False):
        temp_reward_advantage = 0
        temp_cost_advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            # if we are at the last step, bootstrap the return value
            if step == self.num_transitions_per_env - 1:
                next_reward_values = last_reward_values
                next_cost_values = last_cost_values
            else:
                next_reward_values = self.reward_values[step + 1]
                next_cost_values = self.cost_values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            reward_delta = self.rewards[step] + next_is_not_terminal * gamma * next_reward_values - self.reward_values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            temp_reward_advantage = reward_delta + next_is_not_terminal * gamma * lam * temp_reward_advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.reward_targets[step] = temp_reward_advantage + self.reward_values[step]

            cost_delta = self.costs[step] + next_is_not_terminal * gamma * next_cost_values - self.cost_values[step]
            temp_cost_advantage = cost_delta + next_is_not_terminal * gamma * lam * temp_cost_advantage
            self.cost_targets[step] = temp_cost_advantage + self.cost_values[step]

        # Compute the advantages
        self.reward_advantages = self.reward_targets - self.reward_values
        self.cost_advantages = self.cost_targets - self.cost_values

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        prop_obs = self.prop_obs.flatten(0, 1)
        prop_obs_history = self.prop_obs_history.flatten(0, 1)
        actor_obs = self.actor_obs.flatten(0, 1)
        critic_observations = self.critic_obs.flatten(0, 1)

        height_obs = self.height_obs.flatten(0, 1)
        velocity_estimator_obs = self.velocity_estimator_obs.flatten(0, 1)
        velocity_estimator_target = self.velocity_estimator_target.flatten(0, 1)
        priv_obs = self.priv_obs.flatten(0, 1)
        priv_obs_target = self.priv_obs_target.flatten(0, 1)
        next_prop_obs = self.next_prop_obs.flatten(0, 1)

        actions = self.actions.flatten(0, 1)
        reward_values = self.reward_values.flatten(0, 1)
        reward_targets = self.reward_targets.flatten(0, 1)

        # For PPO
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        reward_advantages = self.reward_advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        # For P3O
        cost_values = self.cost_values.flatten(0, 1)
        cost_targets = self.cost_targets.flatten(0, 1)
        cost_advantages = self.cost_advantages.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Create the mini-batch
                # -- Core
                prop_obs_batch = prop_obs[batch_idx]
                next_prop_obs_batch = next_prop_obs[batch_idx]
                prop_obs_history_batch = prop_obs_history[batch_idx]
                actor_obs_batch = actor_obs[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                velocity_estimator_obs_batch = velocity_estimator_obs[batch_idx]
                velocity_estimator_target_obs_batch = velocity_estimator_target[batch_idx]
                priv_obs_batch = priv_obs[batch_idx]
                priv_obs_target_batch = priv_obs_target[batch_idx]
                height_obs_batch = height_obs[batch_idx]
                actions_batch = actions[batch_idx]

                # -- For PPO
                target_values_batch = reward_values[batch_idx]
                reward_targets_batch = reward_targets[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                reward_advantages_batch = reward_advantages[batch_idx]

                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # -- For P3O
                cost_target_values_batch = cost_values[batch_idx]
                cost_targets_batch = cost_targets[batch_idx]
                cost_advantages_batch = cost_advantages[batch_idx]

                yield prop_obs_batch, next_prop_obs_batch, prop_obs_history_batch, actor_obs_batch, critic_observations_batch, \
                    velocity_estimator_obs_batch, velocity_estimator_target_obs_batch, priv_obs_batch, priv_obs_target_batch, height_obs_batch, \
                    actions_batch, target_values_batch, cost_target_values_batch, reward_advantages_batch, cost_advantages_batch, reward_targets_batch, cost_targets_batch, \
                    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None,  # , rnd_state_batch
