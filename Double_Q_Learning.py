import time
import random
import math
import collections
import argparse
from typing import Tuple, List

from MajicWumpusWorld import MagicWumpusWorld

# --- Hyperparameters ---
INITIAL_SEED = 0
TIME_LIMIT_SEC = 44.5
MAX_EPS = 3000
GAMMA_VAL = 1.0
EPS_A = 0.4
EPS_B = 0.02
EPS_DECAY = 800.0
UCB_COEF = 1.5
OPT_INIT = 5.0
TIE_NOISE = 0.2
MIN_LR = 0.02
GOAL_THRESHOLD = -6.0
MOVES = ["Up", "Down", "Left", "Right"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=INITIAL_SEED)
    parser.add_argument('--episodes', type=int, default=MAX_EPS)
    parser.add_argument('--seconds', type=float, default=TIME_LIMIT_SEC)
    parser.add_argument('--optimistic', type=float, default=OPT_INIT)
    args = parser.parse_args()

    env = MagicWumpusWorld()

    returns, td_means, elapsed_time = run_agent(
        env,
        seed=args.seed,
        max_episodes=args.episodes,
        max_seconds=args.seconds,
        optimistic_init=args.optimistic,
    )

    if len(returns) == 0:
        mean_last10 = 0.0
    else:
        last10 = returns[-10:] if len(returns) >= 10 else returns
        mean_last10 = sum(last10) / len(last10)

    print(f"Mean cummulative reward in the last 10 episodes : {mean_last10:.4f}")
    print(f"Time taken by the algorithm : {elapsed_time:.2f} Seconds")


class ActionChooser:
    def __init__(self, moves: List[str], ucb_coef: float, tie_noise: float, optimistic_init: float):
        self.moves = moves
        self.k = len(moves)
        self.ucb_coef = ucb_coef
        self.tie_noise = tie_noise
        self.optimistic_init = optimistic_init

    def _ensure_state_entry(self, q_table, state):
        # Ensure q_table has an entry for `state`.
        if state not in q_table:
            q_table[state] = [self.optimistic_init for _ in range(self.k)]

    def _argmax_random(self, values: List[float]) -> int:
        # Argmax with random tie-breaking.
        m = max(values)
        tied_indices = [i for i, v in enumerate(values) if v == m]
        return random.choice(tied_indices) if len(tied_indices) > 1 else tied_indices[0]

    def choose(self, q_table1, q_table2, state, state_action_counts, epsilon) -> int:
        
        # Choose an action using combined Double-Q estimates plus UCB-style bonus,
        # with epsilon-greedy exploration and occasional tie/noise behavior.
        
        self._ensure_state_entry(q_table1, state)
        self._ensure_state_entry(q_table2, state)

        combined_values = [q_table1[state][a] + q_table2[state][a] for a in range(self.k)]
        total_visits = sum(state_action_counts[(state, a)] for a in range(self.k))

        bonuses = [
            self.ucb_coef * math.sqrt(math.log(1 + max(1, total_visits)) / (1 + state_action_counts[(state, a)]))
            for a in range(self.k)
        ]

        scores = [combined_values[a] + bonuses[a] for a in range(self.k)]

        # Epsilon-greedy
        if random.random() < epsilon:
            return random.randrange(self.k)

        action = self._argmax_random(scores)

        # Small probability of randomness/tie noise
        if random.random() < self.tie_noise:
            if random.random() < 0.15:
                action = random.randrange(self.k)

        return action


def run_agent(env,
              seed: int = INITIAL_SEED,
              max_episodes: int = MAX_EPS,
              max_seconds: float = TIME_LIMIT_SEC,
              optimistic_init: float = OPT_INIT) -> Tuple[List[float], List[float], float]:
    random.seed(seed)
    start_time = time.time()

    action_chooser = ActionChooser(MOVES, UCB_COEF, TIE_NOISE, optimistic_init)

    # Double-Q tables
    q_table_1 = {}
    q_table_2 = {}

    # counts of visits per (state, action)
    state_action_visit_counts = collections.defaultdict(int)

    episode_returns = []
    td_mean_per_episode = []

    for episode in range(1, max_episodes + 1):
        if time.time() - start_time > max_seconds:
            break

        epsilon = EPS_B + (EPS_A - EPS_B) * math.exp(- (episode - 1) / EPS_DECAY)

        env.reset(seed=(seed + episode))
        state = env.CurrentState()
        if state is None:
            state = (1, 1)

        # Ensure starting state exists in both Q-tables
        for table in (q_table_1, q_table_2):
            action_chooser._ensure_state_entry(table, state)

        action = action_chooser.choose(q_table_1, q_table_2, state, state_action_visit_counts, epsilon)

        episode_return = 0.0
        td_total_error = 0.0
        td_step_count = 0

        while True:
            action_chooser._ensure_state_entry(q_table_1, state)
            action_chooser._ensure_state_entry(q_table_2, state)

            reward, next_state = env.TakeAction(MOVES[action])
            episode_return += reward

            state_action_visit_counts[(state, action)] += 1
            learning_rate = max(MIN_LR, 1.0 / (1.0 + state_action_visit_counts[(state, action)]))

            if random.random() < 0.5:
                # Update q_table_1 using q_table_2 for target
                if next_state is not None:
                    action_chooser._ensure_state_entry(q_table_1, next_state)
                    action_chooser._ensure_state_entry(q_table_2, next_state)
                    next_action = action_chooser._argmax_random(q_table_1[next_state])
                    target = reward + GAMMA_VAL * q_table_2[next_state][next_action]
                else:
                    target = reward
                delta = target - q_table_1[state][action]
                q_table_1[state][action] += learning_rate * delta
            else:
                # Update q_table_2 using q_table_1 for target
                if next_state is not None:
                    action_chooser._ensure_state_entry(q_table_1, next_state)
                    action_chooser._ensure_state_entry(q_table_2, next_state)
                    next_action = action_chooser._argmax_random(q_table_2[next_state])
                    target = reward + GAMMA_VAL * q_table_1[next_state][next_action]
                else:
                    target = reward
                delta = target - q_table_2[state][action]
                q_table_2[state][action] += learning_rate * delta

            td_total_error += abs(delta)
            td_step_count += 1

            if next_state is None:
                break

            state = next_state
            action = action_chooser.choose(q_table_1, q_table_2, state, state_action_visit_counts, epsilon)

        episode_returns.append(episode_return)
        td_mean_per_episode.append(td_total_error / td_step_count if td_step_count > 0 else 0.0)

        if len(episode_returns) >= 10:
            last10_mean = sum(episode_returns[-10:]) / 10.0
            if last10_mean >= GOAL_THRESHOLD - 1e-9:
                break

    elapsed_time = time.time() - start_time
    return episode_returns, td_mean_per_episode, elapsed_time


if __name__ == '__main__':
    main()
