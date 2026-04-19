"""
Multi-restart Simulated Annealing for inventory-management policy optimization.
Perturbs neural-network policy weights, accepts via Metropolis criterion,
restarts with explore/exploit scheduling under a fixed episode/time budget.
"""

import copy
import time

import numpy as np
import torch

from common import PolicyNetwork
from ML4CE_RL_environment import MESCEnv
from utils import setup_model_saving


def _perturb_weights(params, sigma=0.1):
    """Add Gaussian noise to each NN weight tensor."""
    new_params = {}
    for k, v in params.items():
        new_params[k] = v + torch.randn(v.shape) * sigma
    return new_params


def _run_episodes(policy_net, env, n=5):
    """Run n episodes and return mean and std of episode reward."""
    rewards = []

    for _ in range(n):
        env.reset()
        state = env.state
        done = False
        ep_reward = 0

        while not done:
            with torch.no_grad():
                action = policy_net(torch.FloatTensor(state))
                action = np.clip(np.fix(action.numpy()), 0, env.action_space.high)
            state, r, done, _ = env.step(action)
            ep_reward += r

        rewards.append(ep_reward)

    return np.mean(rewards), np.std(rewards)


def your_optimization_alg(
    env: MESCEnv,
    policy_net: PolicyNetwork,
    *,
    max_episodes=5000,
    max_time=5 * 60,
):
    """Multi-restart SA over policy-network weights. Returns (best_state_dict, plot_data)."""
    save_f_path = setup_model_saving(algorithm="Your algorithm")

    plot_data = {"reward_history": [], "episodes": [], "std_history": []}

    start_time = time.time()
    best_reward = -np.inf
    best_policy = policy_net.state_dict()

    num_episodes_avg = 5
    initial_temp = 5e5
    param_max = 0.4
    num_restarts = 10

    overall_best_reward = -np.inf
    overall_best_param = None
    total_episodes = 0
    episodes_per_restart = max_episodes // num_restarts

    for restart in range(num_restarts):

        if time.time() - start_time > max_time:
            break

        # Early restarts explore (random init), later restarts exploit (perturb best)
        explore_prob = 0.8 - (restart / num_restarts) * 0.6
        do_explore = np.random.rand() < explore_prob

        if restart == 0:
            pass

        elif not do_explore and overall_best_param is not None:
            policy_net.load_state_dict(overall_best_param)
            current_param = _perturb_weights(overall_best_param, sigma=0.05)
            print(f"Restart {restart + 1}: Exploiting best (p_explore={explore_prob:.2f})")

        else:
            print(f"Restart {restart + 1}: Exploring (p_explore={explore_prob:.2f})")
            for p in policy_net.parameters():
                if len(p.shape) >= 2:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    torch.nn.init.zeros_(p)

        current_param = copy.deepcopy(policy_net.state_dict())
        restart_best_param = copy.deepcopy(current_param)

        policy_net.load_state_dict(current_param)
        current_reward, std = _run_episodes(policy_net, env, num_episodes_avg)
        restart_best_reward = current_reward
        total_episodes += num_episodes_avg

        max_iter = int((episodes_per_restart - num_episodes_avg) / num_episodes_avg)
        i = 0

        for i in range(max_iter):
            if time.time() - start_time > max_time:
                break

            # Step size shrinks with progress within a restart
            progress = i / max(max_iter, 1)
            sigma = param_max * (1 - 0.5 * progress)

            candidate_param = _perturb_weights(current_param, sigma=sigma)
            policy_net.load_state_dict(candidate_param)
            candidate_reward, std = _run_episodes(policy_net, env, num_episodes_avg)

            if candidate_reward > restart_best_reward:
                restart_best_reward = candidate_reward
                restart_best_param = copy.deepcopy(candidate_param)

            # Metropolis acceptance with cooling temperature
            delta = candidate_reward - current_reward
            temp = initial_temp / (1 + i * 0.5)

            if delta > 0 or np.random.rand() < np.exp(delta / temp):
                current_param = copy.deepcopy(candidate_param)
                current_reward = candidate_reward

            total_episodes += num_episodes_avg
            plot_data['episodes'].append(total_episodes)
            plot_data['reward_history'].append(candidate_reward)
            plot_data['std_history'].append(std)

        if restart_best_reward > overall_best_reward:
            overall_best_reward = restart_best_reward
            overall_best_param = copy.deepcopy(restart_best_param)

        if restart_best_reward > best_reward:
            best_reward = restart_best_reward
            best_policy = copy.deepcopy(restart_best_param)

        print(f"Restart {restart + 1}: Best = {restart_best_reward:.2f} (iters={i + 1})")

        if (time.time() - start_time) > max_time:
            print("Timeout reached: the best policy found so far will be returned.")
            break

    print(f"Policy model weights saved in: {save_f_path}")
    print(f"Best reward: {best_reward}")

    return best_policy, plot_data
