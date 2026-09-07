from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules.auxiliary_networks import AuxiliaryNetworks
from rsl_rl.modules.actor_critic import Actor, RewardCritic, CostCritic
from rsl_rl.modules.normalizer import EmpiricalNormalization, EmpiricalHistoryNormalization


class OnPolicyRunner:
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.aux_networks_cfg = train_cfg["aux_network"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.num_cost = self.cfg["num_cost"]

        # resolve dimensions of observations
        _, obs_dict = self.env.get_observations()
        num_actor_obs = obs_dict["observations"]["policy"].shape[1]
        num_critic_obs = obs_dict["observations"]["critic_obs"].shape[1]
        num_prop_obs = obs_dict["observations"]["prop_obs"].shape[1]
        num_prop_obs_history = obs_dict["observations"]["prop_obs_history"].shape[1]
        num_velocity_estimator_obs = obs_dict["observations"]["velocity_estimator_obs"].shape[1]
        num_velocity_estimator_target = obs_dict["observations"]["velocity_estimator_target"].shape[1]
        num_priv_obs = obs_dict["observations"]["priv_obs"].shape[1]
        num_priv_obs_target = obs_dict["observations"]["priv_obs_target"].shape[1]
        num_height_scan = obs_dict["observations"]["height_obs"].shape[1]
        # num_height_scan = 289

        num_velocity_estimator_obs = num_prop_obs_history  # velocity estimator input은 history로 들어감
        num_priv_obs = num_prop_obs_history  # privileged obs input도 history로 들어감

        actor_class = Actor
        reward_critic_class = RewardCritic
        cost_critic_class = CostCritic

        actor = actor_class(num_actor_obs, self.env.num_actions, **self.policy_cfg).to(self.device)
        rewardCritic = reward_critic_class(num_critic_obs, **self.policy_cfg).to(self.device)
        costCritic = cost_critic_class(num_critic_obs, self.num_cost, **self.policy_cfg).to(self.device)

        aux_nets_class = AuxiliaryNetworks
        self.aux_nets: AuxiliaryNetworks = aux_nets_class(num_velocity_estimator_obs, num_priv_obs, \
                                                          num_height_scan, num_prop_obs, num_prop_obs_history, 16, **self.aux_networks_cfg)

        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        self.alg_cfg.pop("symmetry_cfg", None)
        self.alg_cfg.pop("rnd_cfg", None)
        self.alg: PPO = alg_class(actor, rewardCritic, costCritic, self.aux_nets, self.num_cost, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # self.obs_history_normalizer = EmpiricalNormalization(shape=[num_prop_obs], until=1.0e8).to(self.device)
        # self.velocity_obs_normalizer = EmpiricalNormalization(shape=[num_velocity_estimator_target], until=1.0e8).to(self.device)
        # self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_priv_obs_target], until=1.0e8).to(self.device)

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_cost,
            self.num_steps_per_env,
            [num_prop_obs],
            [num_prop_obs_history],
            [num_actor_obs],
            [num_critic_obs],
            [num_height_scan],
            [num_velocity_estimator_obs],
            [num_velocity_estimator_target],
            [num_priv_obs],
            [num_priv_obs_target],
            [self.env.num_actions],
        )
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]
        self.dagger_update_freq = 20

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()
            self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        # start learning
        _, obs_dict = self.env.get_observations()
        actor_obs = obs_dict["observations"]["policy"].to(self.device)
        critic_obs = obs_dict["observations"]["critic_obs"].to(self.device)
        prop_obs = obs_dict["observations"]["prop_obs"].to(self.device)
        prop_obs_history = obs_dict["observations"]["prop_obs_history"].to(self.device)
        velocity_estimator_obs = obs_dict["observations"]["velocity_estimator_obs"].to(self.device)
        velocity_estimator_target = obs_dict["observations"]["velocity_estimator_target"].to(self.device)
        priv_obs = obs_dict["observations"]["priv_obs"].to(self.device)
        priv_obs_target = obs_dict["observations"]["priv_obs_target"].to(self.device)
        height_obs = obs_dict["observations"]["height_obs"].to(self.device)

        actor_obs, critic_obs = actor_obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        command_infos = []
        cost_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start_iter_t = time.time()
            act_time = 0.0
            step_time = 0.0
            proc_time = 0.0

            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(prop_obs,
                                           prop_obs_history,
                                           critic_obs,
                                           prop_obs_history,  # velocity_estimator_obs
                                           velocity_estimator_target,
                                           prop_obs_history,  # priv_obs
                                           priv_obs_target,
                                           height_obs)

                    obs_dict, rewards, costs, dones, extras = self.env.step(actions.to(self.device))
                    rewards, costs, dones = rewards.to(self.device), costs.to(self.device), dones.to(self.device)
                    self.alg.transition.next_prop_observations = extras["observations"]["prop_obs"].to(self.device)

                    prop_obs = extras["observations"]["prop_obs"].to(self.device)
                    prop_obs_history = extras["observations"]["prop_obs_history"].to(self.device)
                    height_obs = extras["observations"]["height_obs"].to(self.device)
                    critic_obs = extras["observations"]["critic_obs"].to(self.device)
                    velocity_estimator_target = extras["observations"]["velocity_estimator_target"].to(self.device)
                    priv_obs_target = extras["observations"]["priv_obs_target"].to(self.device)

                    # Process env step and store in buffer
                    self.alg.process_env_step(rewards, costs, dones, extras)

                    if self.log_dir is not None:
                        # Book keeping
                        if "Episode_Reward" in extras["log"]:
                            ep_infos.append(extras["log"]["Episode_Reward"])
                        if "Command" in extras["log"]:
                            command_infos.append(extras["log"]["Command"])

                        cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        # rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()) # 나중에 대거 사용시 필요
                        # lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                stop = time.time()
                collection_time = stop - start_iter_t

                # Learning step
                start = time.time()
                self.alg.compute_returns(critic_obs)

            loss_dict, cost_returns_per_type = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        ep_string = self.log_info_block(locs["ep_infos"], "Episode", self.writer, self.device, locs["it"], pad)
        ep_string += self.log_info_block(locs["command_infos"], "Command", self.writer, self.device, locs["it"], pad)
        ep_string += self.log_info_block(locs["cost_infos"], "Cost", self.writer, self.device, locs["it"], pad)

        mean_std = self.alg.actor.action_std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))
        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        if "cost_returns_per_type" in locs:
            for i in range(locs["cost_returns_per_type"].shape[0]):
                self.writer.add_scalar(f"CostReturn/cost_{i}", locs["cost_returns_per_type"][i].item(), locs["it"])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
            self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            # -- Losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""

        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                               locs['start_iter'] + locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor.state_dict(),    
            "optimizer_state_dict": self.alg.actor_optimizer.state_dict(),
            "reward_critic_state_dict": self.alg.reward_critic.state_dict(),
            "reward_critic_optimizer_state_dict": self.alg.reward_critic_optimizer.state_dict(),
            "cost_critic_model_state_dict": self.alg.cost_critic.state_dict(),
            "cost_critic_optimizer_state_dict": self.alg.cost_critic_optimizer.state_dict(),
            "aux_net_state_dict": self.alg.aux_networks.state_dict(),
            "aux_net_optimizer_state_dict": self.alg.aux_networks_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # saved_dict["obs_history_norm_state_dict"] = self.obs_history_normalizer.state_dict()
        # saved_dict["vel_norm_state_dict"] = self.velocity_obs_normalizer.state_dict()
        # saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path)
        self.alg.actor.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.reward_critic.load_state_dict(loaded_dict["reward_critic_state_dict"])
        self.alg.cost_critic.load_state_dict(loaded_dict["cost_critic_model_state_dict"])
        self.alg.aux_networks.load_state_dict(loaded_dict["aux_net_state_dict"])
        # -----------------load normalizer----------------------------- #
        # self.obs_history_normalizer.load_state_dict(loaded_dict["obs_history_norm_state_dict"])
        # self.velocity_obs_normalizer.load_state_dict(loaded_dict["vel_norm_state_dict"])
        # self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])

        if load_optimizer:
            self.alg.actor_optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.alg.reward_critic_optimizer.load_state_dict(loaded_dict["reward_critic_optimizer_state_dict"])
            self.alg.cost_critic_optimizer.load_state_dict(loaded_dict["cost_critic_optimizer_state_dict"])
            self.alg.aux_networks_optimizer.load_state_dict(loaded_dict["aux_net_optimizer_state_dict"])

        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor.to(device)
        policy = self.alg.actor.act_inference
        return policy

    def train_mode(self):
        # -- PPO
        self.alg.actor.train()
        self.alg.reward_critic.train()
        self.alg.cost_critic.train()
        # self.obs_history_normalizer.train()
        # self.privileged_obs_normalizer.train()
        # self.velocity_obs_normalizer.train()

    def eval_mode(self):
        # -- PPO
        self.alg.actor.eval()
        self.alg.reward_critic.eval()
        self.alg.cost_critic.eval()
        # self.obs_history_normalizer.train()
        # self.privileged_obs_normalizer.train()
        # self.velocity_obs_normalizer.train()

    def log_info_block(self, infos, prefix, writer, device, it, pad):
        log_string = ""
        if infos:
            keys = set().union(*[info.keys() for info in infos])
            for key in keys:
                infotensor = torch.tensor([], device=device)
                for info in infos:
                    if key not in info:
                        continue
                    val = info[key]
                    if not isinstance(val, torch.Tensor):
                        val = torch.Tensor([val])
                    if len(val.shape) == 0:
                        val = val.unsqueeze(0)
                    infotensor = torch.cat((infotensor, val.to(device)))
                if infotensor.numel() == 0:
                    continue
                value = torch.mean(infotensor)
                writer.add_scalar(f"{prefix}/{key}", value, it)
                log_string += f"""{f'Mean {prefix} {key}:':>{pad}} {value:.4f}\n"""
        return log_string
