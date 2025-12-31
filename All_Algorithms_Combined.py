#!/usr/bin/env python3
"""
ROLLNO_NAME.py

Implements baseline TD control algorithms (SARSA, Q-learning, Expected SARSA)
and an improved Double Q variant. Runs controlled experiments in the
Magic Wumpus World and produces plots (regret, avg TD error) and prints
final outputs required by the assignment:

Mean cummulative reward in the last 10 episodes : <value>
Time taken by the algorithm : <seconds> Seconds

Usage examples:
  python ROLLNO_NAME.py --algo improved        # run the improved double-Q
  python ROLLNO_NAME.py --algo sarsa --plot    # run SARSA and save plots
  python ROLLNO_NAME.py --algo all --plot      # run all baselines (careful with time)

Note: keep this file alongside MagicWumpusWorld.py and MWW.json.
"""

import time
import random
import math
import collections
import argparse
import csv
import os
from typing import Tuple, List

# plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Default hyperparameters ---
SEED = 0
MAX_SECONDS = 44.5
MAX_EPISODES = 3000
GAMMA = 1.0
EPS_START = 0.41
EPS_END = 0.02
EPS_DECAY_EPISODES = 800.0
UCB_C = 1.5
OPTIMISTIC_INIT = 5.0
RANDOM_ACTION_PROB_ON_TIED = 0.2
ALPHA_MIN = 0.02
EARLY_STOP_TARGET = -6.0  # optimal return (environment's shortest safe path)
ACTIONS = ["Up", "Down", "Left", "Right"]

# --- Utilities ---

def argmax_with_random_tie(values: List[float]) -> int:
    maxv = max(values)
    indices = [i for i, v in enumerate(values) if v == maxv]
    return random.choice(indices) if len(indices) > 1 else indices[0]

# --- Environment import (must exist in same folder) ---
try:
    from MajicWumpusWorld import MagicWumpusWorld
except Exception:
    # Some systems might have the file named MagicWumpusWorld.py
    try:
        from MagicWumpusWorld import MagicWumpusWorld
    except Exception:
        raise

# --- Generic TD runner used by all methods ---

def make_policy_from_Q_eps_greedy(Q_s: List[float], eps: float) -> List[float]:
    """Return probability distribution over actions for epsilon-greedy."""
    k = len(Q_s)
    greedy_idx = argmax_with_random_tie(Q_s)
    probs = [eps / k for _ in range(k)]
    probs[greedy_idx] += 1.0 - eps
    return probs


def select_action(Q1, Q2, s, eps, use_ucb, N_sa, total_sa_visits):
    # ensure Q entries exist will be handled by caller
    # Combined Q used for greedy selection: sum of both tables
    combined = [Q1[s][a] + Q2[s][a] for a in range(len(ACTIONS))]
    if use_ucb:
        bonuses = []
        # small safeguard: total_sa_visits should be >=1 to avoid log(0)
        base = max(1, total_sa_visits)
        for a in range(len(ACTIONS)):
            bonus = UCB_C * math.sqrt(math.log(base + 1) / (1 + N_sa[(s, a)]))
            bonuses.append(bonus)
        scores = [combined[a] + bonuses[a] for a in range(len(ACTIONS))]
        a = argmax_with_random_tie(scores)
    else:
        # epsilon-greedy on combined estimate
        if random.random() < eps:
            a = random.randrange(len(ACTIONS))
        else:
            a = argmax_with_random_tie(combined)
            if random.random() < RANDOM_ACTION_PROB_ON_TIED:
                if random.random() < 0.15:
                    a = random.randrange(len(ACTIONS))
    return a


def ensure_state(Q, s, optimistic_init=OPTIMISTIC_INIT):
    if s not in Q:
        Q[s] = [optimistic_init for _ in ACTIONS]


# --- Implementations of algorithms ---
def run_algorithm(env, algo: str, seed: int = SEED, max_episodes: int = MAX_EPISODES,
                  max_seconds: float = MAX_SECONDS, optimistic_init: float = OPTIMISTIC_INIT,
                  use_ucb: bool = False) -> Tuple[List[float], List[float], float]:
    """
    Run one instance of the chosen algorithm and return (returns, avg_td_per_episode, time_taken)
    algo in {"sarsa", "qlearning", "expected_sarsa", "improved"}
    """
    random.seed(seed)
    start_time = time.time()

    Q1 = {}
    Q2 = {}
    # For single-table methods, we'll use Q1 and never touch Q2
    N_sa = collections.defaultdict(int)
    N_s_total = collections.defaultdict(int)

    returns = []
    td_avgs = []

    total_steps = 0

    for ep in range(1, max_episodes + 1):
        # time check
        if time.time() - start_time > max_seconds:
            break

        eps = EPS_END + (EPS_START - EPS_END) * math.exp(- (ep-1) / EPS_DECAY_EPISODES)

        env.reset(seed=(seed + ep))
        s = env.CurrentState()
        if s is None:
            s = (1,1)

        ep_reward = 0.0
        td_sum = 0.0
        td_count = 0

        # ensure starting state exists
        ensure_state(Q1, s, optimistic_init)
        ensure_state(Q2, s, optimistic_init)
        N_s_total[s] += 1

        # for SARSA we need to select initial action under policy
        if algo == 'sarsa':
            # select action from Q1 (we use Q1 as main table)
            if use_ucb:
                total_visits = sum(N_sa[(s,a)] for a in range(len(ACTIONS)))
                a = select_action(Q1, Q2, s, eps, True, N_sa, total_visits)
            else:
                # epsilon-greedy on Q1
                if random.random() < eps:
                    a = random.randrange(len(ACTIONS))
                else:
                    a = argmax_with_random_tie(Q1[s])
        else:
            a = None

        while True:
            ensure_state(Q1, s, optimistic_init)
            ensure_state(Q2, s, optimistic_init)
            N_s_total[s] += 0  # already incremented above for start; keep for completeness

            # determine selection method for this step
            total_sa_visits = sum(N_sa[(s,aa)] for aa in range(len(ACTIONS)))

            if algo == 'sarsa':
                # use previously selected 'a'
                pass
            else:
                # for q-learning and expected sarsa and improved double q use selection via combined
                if use_ucb:
                    a = select_action(Q1, Q2, s, eps, True, N_sa, total_sa_visits)
                else:
                    if random.random() < eps:
                        a = random.randrange(len(ACTIONS))
                    else:
                        # for single-table methods use combined Q (Q1+Q2) for greedy selection
                        combined = [Q1[s][aa] + Q2[s][aa] for aa in range(len(ACTIONS))]
                        a = argmax_with_random_tie(combined)

            action = ACTIONS[a]
            reward, next_state = env.TakeAction(action)
            ep_reward += reward
            total_steps += 1

            # update counts
            N_sa[(s, a)] += 1

            # adaptive alpha
            alpha = max(ALPHA_MIN, 1.0 / (1.0 + N_sa[(s, a)]))

            # compute target according to algorithm
            if algo == 'sarsa':
                # observe next action under policy
                if next_state is not None:
                    ensure_state(Q1, next_state, optimistic_init)
                    ensure_state(Q2, next_state, optimistic_init)
                    if use_ucb:
                        next_total = sum(N_sa[(next_state,aa)] for aa in range(len(ACTIONS)))
                        a_next = select_action(Q1, Q2, next_state, eps, True, N_sa, next_total)
                    else:
                        if random.random() < eps:
                            a_next = random.randrange(len(ACTIONS))
                        else:
                            a_next = argmax_with_random_tie(Q1[next_state])
                    target = reward + (GAMMA * Q1[next_state][a_next])
                else:
                    target = reward
                delta = target - Q1[s][a]
                Q1[s][a] += alpha * delta

                # move
                if next_state is None:
                    td_sum += abs(delta); td_count += 1
                    break
                s = next_state
                a = a_next

            elif algo == 'qlearning':
                if next_state is not None:
                    ensure_state(Q1, next_state, optimistic_init)
                    # off-policy max
                    a_star = argmax_with_random_tie(Q1[next_state])
                    target = reward + (GAMMA * Q1[next_state][a_star])
                else:
                    target = reward
                delta = target - Q1[s][a]
                Q1[s][a] += alpha * delta

                td_sum += abs(delta); td_count += 1
                if next_state is None:
                    break
                s = next_state

            elif algo == 'expected_sarsa':
                if next_state is not None:
                    ensure_state(Q1, next_state, optimistic_init)
                    # compute expected Q under eps-greedy policy derived from Q1
                    probs = make_policy_from_Q_eps_greedy(Q1[next_state], eps)
                    expected = sum(p * Q1[next_state][aa] for aa, p in enumerate(probs))
                    target = reward + (GAMMA * expected)
                else:
                    target = reward
                delta = target - Q1[s][a]
                Q1[s][a] += alpha * delta

                td_sum += abs(delta); td_count += 1
                if next_state is None:
                    break
                s = next_state

            elif algo == 'improved':
                # improved double Q variant with optional UCB-aware argmax for targets
                # randomly pick which Q to update
                if random.random() < 0.5:
                    # update Q1 using argmax of Q1 (or UCB) and value from Q2
                    ensure_state(Q1, s, optimistic_init)
                    if next_state is not None:
                        ensure_state(Q1, next_state, optimistic_init)
                        ensure_state(Q2, next_state, optimistic_init)

                        if use_ucb:
                            next_total = sum(N_sa[(next_state,aa)] for aa in range(len(ACTIONS)))
                            # select_action returns the action index possibly using UCB bonuses
                            a_prime = select_action(Q1, Q2, next_state, eps, True, N_sa, next_total)
                        else:
                            # standard double-Q: argmax over Q1
                            a_prime = argmax_with_random_tie(Q1[next_state])

                        target = reward + GAMMA * Q2[next_state][a_prime]
                    else:
                        target = reward
                    delta = target - Q1[s][a]
                    Q1[s][a] += alpha * delta
                else:
                    ensure_state(Q2, s, optimistic_init)
                    if next_state is not None:
                        ensure_state(Q1, next_state, optimistic_init)
                        ensure_state(Q2, next_state, optimistic_init)

                        if use_ucb:
                            next_total = sum(N_sa[(next_state,aa)] for aa in range(len(ACTIONS)))
                            a_prime = select_action(Q1, Q2, next_state, eps, True, N_sa, next_total)
                        else:
                            a_prime = argmax_with_random_tie(Q2[next_state])

                        target = reward + GAMMA * Q1[next_state][a_prime]
                    else:
                        target = reward
                    delta = target - Q2[s][a]
                    Q2[s][a] += alpha * delta

                td_sum += abs(delta); td_count += 1
                if next_state is None:
                    break
                s = next_state

            else:
                raise ValueError(f"Unknown algo: {algo}")

        # episode finished
        returns.append(ep_reward)
        episode_td_avg = (td_sum / td_count) if td_count > 0 else 0.0
        td_avgs.append(episode_td_avg)

        # simple early-stop: last 10 mean near optimal
        if len(returns) >= 10:
            last10_mean = sum(returns[-10:]) / 10.0
            if last10_mean >= EARLY_STOP_TARGET - 1e-9:
                break

    time_taken = time.time() - start_time
    return returns, td_avgs, time_taken


# --- Plotting helpers ---
def save_plots(all_results: dict, out_prefix: str = 'results', smooth_w: int = 5, smooth_method: str = 'gaussian'):
    """
    Save individual plots per-algorithm AND combined plots that overlay
    all algorithms for easier comparison.

    Creates files under ./plots:
      - {out_prefix}_{name}_regret.png, _td.png, _returns.png  (per-algo)
      - {out_prefix}_combined_regret.png
      - {out_prefix}_combined_td.png
      - {out_prefix}_combined_returns.png

    Parameters:
      smooth_w: integer moving-window size (1 disables smoothing)
      smooth_method: 'moving' for moving-average, 'exp' for exponential smoothing, 'gaussian' for gaussian kernel
    """
    os.makedirs('plots', exist_ok=True)

    import numpy as _np

    def _gaussian_kernel(w):
        if w <= 1:
            return _np.array([1.0])
        # prefer odd window
        if w % 2 == 0:
            w = w + 1
        x = _np.arange(w) - (w // 2)
        sigma = max(0.5, float(w) / 6.0)  # heuristic: window ~ 6 sigma
        kernel = _np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        return kernel

    def _apply_smoothing(arr):
        """Return smoothed numpy array (same length) according to chosen method."""
        if arr is None:
            return arr
        if smooth_w is None or smooth_w <= 1:
            return _np.array(arr, dtype=float)
        a = _np.array(arr, dtype=float)
        if a.size == 0:
            return a
        w = int(smooth_w)
        if smooth_method == 'moving':
            if w >= a.size:
                return _np.ones_like(a) * _np.mean(a)
            kernel = _np.ones(w) / float(w)
            smooth = _np.convolve(a, kernel, mode='same')
            return smooth
        elif smooth_method == 'exp':
            alpha = 2.0 / (float(w) + 1.0)
            smooth = _np.empty_like(a)
            smooth[0] = a[0]
            for i in range(1, a.size):
                smooth[i] = alpha * a[i] + (1.0 - alpha) * smooth[i-1]
            return smooth
        elif smooth_method == 'gaussian':
            kernel = _gaussian_kernel(w)
            smooth = _np.convolve(a, kernel, mode='same')
            # for very short arrays, fall back to mean
            if a.size < 3:
                return _np.ones_like(a) * _np.mean(a)
            return smooth
        else:
            return a

    # First, save per-algo plots (show raw and smoothed)
    for name, (returns, td_avgs, *_rest) in all_results.items():
        if len(returns) == 0:
            episodes = []
        else:
            episodes = list(range(1, len(returns) + 1))

        # regret per episode, with optimal_return = EARLY_STOP_TARGET
        optimal = EARLY_STOP_TARGET
        regret = [optimal - r for r in returns]

        # apply smoothing
        regret_s = _apply_smoothing(regret)
        td_s = _apply_smoothing(td_avgs)
        returns_s = _apply_smoothing(returns)

        if len(episodes) > 0:
            # Regret: raw (faint) + smoothed (solid)
            plt.figure()
            plt.plot(episodes, regret, alpha=0.25, linewidth=1, label=f'{name} regret (raw)')
            plt.plot(episodes, regret_s, linewidth=2, label=f'{name} regret (smoothed)')
            plt.xlabel('Episode')
            plt.ylabel('Regret per episode')
            plt.title(f'Regret per episode - {name}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'plots/{out_prefix}_{name}_regret.png')
            plt.close()

            # Avg TD-error magnitude
            plt.figure()
            plt.plot(episodes, td_avgs, alpha=0.25, linewidth=1, label=f'{name} avg |TD| (raw)')
            plt.plot(episodes, td_s, linewidth=2, label=f'{name} avg |TD| (smoothed)')
            plt.xlabel('Episode')
            plt.ylabel('Average TD-error magnitude')
            plt.title(f'Average TD-error magnitude per episode - {name}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'plots/{out_prefix}_{name}_td.png')
            plt.close()

            # Returns
            plt.figure()
            plt.plot(episodes, returns, alpha=0.25, linewidth=1, label=f'{name} returns (raw)')
            plt.plot(episodes, returns_s, linewidth=2, label=f'{name} returns (smoothed)')
            plt.xlabel('Episode')
            plt.ylabel('Episode return')
            plt.title(f'Episode return - {name}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'plots/{out_prefix}_{name}_returns.png')
            plt.close()

    # --- Combined plots: overlay all algorithms on the same axes ---
    # Determine common x-axis length (max episodes among runs)
    max_len = 0
    for (returns, td_avgs, *_rest) in all_results.values():
        if len(returns) > max_len:
            max_len = len(returns)

    if max_len == 0:
        return

    episodes = list(range(1, max_len + 1))

    # Common plotting params
    plt.rcParams.update({'font.size': 12})

    # Combined regret (smoothed)
    plt.figure(figsize=(14, 6))
    for idx, (name, (returns, _td_avgs, *_rest)) in enumerate(all_results.items()):
        if len(returns) == 0:
            padded = [0] * max_len
        else:
            padded = list(returns) + ([returns[-1]] * (max_len - len(returns)))
        optimal = EARLY_STOP_TARGET
        regret = _np.array([optimal - r for r in padded], dtype=float)
        smooth = _apply_smoothing(regret)
        plt.plot(episodes, smooth, linewidth=2, label=name)

    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Regret per Episode', fontsize=14)
    plt.title('Regret per episode - All algorithms', fontsize=18, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'plots/{out_prefix}_combined_regret.png')
    plt.close()

    # Combined avg TD (smoothed)
    plt.figure(figsize=(14, 6))
    for idx, (name, (_returns, td_avgs, *_rest)) in enumerate(all_results.items()):
        if len(td_avgs) == 0:
            padded = [0] * max_len
        else:
            padded = list(td_avgs) + ([td_avgs[-1]] * (max_len - len(td_avgs)))
        arr = _np.array(padded, dtype=float)
        smooth = _apply_smoothing(arr)
        plt.plot(episodes, smooth, linewidth=2, label=name)

    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Average TD-error magnitude', fontsize=14)
    plt.title('Average TD-error magnitude per episode - All algorithms', fontsize=18, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'plots/{out_prefix}_combined_td.png')
    plt.close()

    # Combined returns (smoothed)
    plt.figure(figsize=(14, 6))
    for idx, (name, (returns, _td, *_rest)) in enumerate(all_results.items()):
        if len(returns) == 0:
            padded = [0] * max_len
        else:
            padded = list(returns) + ([returns[-1]] * (max_len - len(returns)))
        arr = _np.array(padded, dtype=float)
        smooth = _apply_smoothing(arr)
        plt.plot(episodes, smooth, linewidth=2, label=name)

    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Episode return', fontsize=14)
    plt.title('Episode return - All algorithms', fontsize=18, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'plots/{out_prefix}_combined_returns.png')
    plt.close()


# --- CSV export ---

def save_csv(name, returns, td_avgs):
    os.makedirs('logs', exist_ok=True)
    with open(f'logs/{name}_metrics.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['episode', 'return', 'avg_td'])
        for i, (r, td) in enumerate(zip(returns, td_avgs), start=1):
            w.writerow([i, r, td])


# --- Main driver ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='improved',
                        choices=['sarsa', 'qlearning', 'expected_sarsa', 'improved', 'all'],
                        help='Algorithm to run')
    parser.add_argument('--plot', action='store_true', help='Save plots')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--episodes', type=int, default=MAX_EPISODES)
    parser.add_argument('--seconds', type=float, default=MAX_SECONDS)
    parser.add_argument('--ucb', action='store_true', help='Use UCB-style exploration')
    parser.add_argument('--optimistic', type=float, default=OPTIMISTIC_INIT)
    parser.add_argument('--smooth-window', type=int, default=80, help='Smoothing window (1 to disable smoothing)')
    parser.add_argument('--smooth-method', choices=['moving','exp','gaussian'], default='gaussian', help='Smoothing method for plots: moving, exp, gaussian')
    args = parser.parse_args()

    env = MagicWumpusWorld()

    to_run = [args.algo] if args.algo != 'all' else ['sarsa', 'qlearning', 'expected_sarsa', 'improved']

    results = {}
    total_start = time.time()

    for name in to_run:
        # keep remaining time budget conservative: subtract elapsed
        elapsed = time.time() - total_start
        remaining = max(1.0, args.seconds - elapsed)
        print(f"Running {name} (seed={args.seed}) with time budget {remaining:.1f}s")
        returns, td_avgs, t_taken = run_algorithm(env, name, seed=args.seed,
                                                  max_episodes=args.episodes,
                                                  max_seconds=remaining,
                                                  optimistic_init=args.optimistic,
                                                  use_ucb=args.ucb)
        results[name] = (returns, td_avgs, t_taken)
        save_csv(name, returns, td_avgs)

    total_time = time.time() - total_start

    # By assignment, print the final output for the improved algorithm (the one we will submit)
    # If user asked a different algo, print that algorithm's final numbers instead.
    primary = 'improved' if args.algo == 'improved' or args.algo == 'all' else args.algo
    prim_returns, prim_td, prim_time = results[primary]

    # mean cumulative reward in last 10 episodes (or fewer)
    if len(prim_returns) == 0:
        mean_last10 = 0.0
    else:
        last10 = prim_returns[-10:] if len(prim_returns) >= 10 else prim_returns
        mean_last10 = sum(last10) / len(last10)

    # Print EXACT required format. Use 4 decimal places for the reward as requested.
    # NOTE: spec uses "cummulative" spelling; we match that.
    print(f"Mean cummulative reward in the last 10 episodes : {mean_last10:.4f}")
    print(f"Time taken by the algorithm : {prim_time:.2f} Seconds")

    if args.plot:
        save_plots(results, out_prefix='mww', smooth_w=args.smooth_window, smooth_method=args.smooth_method)
        print('Plots saved to ./plots, CSV logs to ./logs')


if __name__ == '__main__':
    main()
