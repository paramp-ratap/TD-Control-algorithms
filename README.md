# Magic Wumpus World (TD-control algorithms)

## Overview

This repository contains the Magic Wumpus World (MWW). The environment is a fixed 4×4 grid where an agent must discover a *safe path* from the start cell A (1,1) to the goal G (4,4) while avoiding a stochastic Wumpus hazard. The goal of the assignment is to implement and compare TD-control algorithms, propose an improved TD variant, and produce a report with experimental analysis.

---

## Environment Details

### Grid and Coordinates

* Grid: fixed 4×4.
* Coordinates are 1-indexed as `(row, col)`.
* Bottom-left cell `(1,1)` is **A** (start).
* Top-right cell `(4,4)` is **G** (goal).
* Moving **Up** increases the row; moving **Right** increases the column.

### Actions

* At each time step the agent selects one of four actions: `Up`, `Down`, `Left`, `Right`.
* Transitions are deterministic.
* If an action would move the agent off the grid, the agent stays in place (invalid move).

### Wumpus Hazard

* Entering any cell **not on the designated safe path** triggers a Bernoulli trial with probability `p` (from `MWW.json`).
* If the Wumpus appears (trial success), the agent **dies** and the episode terminates immediately.
* Cells on the safe path have `p = 0` (Wumpus never appears there).

### Safe Path

* Exactly one contiguous 4-neighbour path from `A` to `G` is marked safe.
* `MWW.json` contains the `safe_path` array of 1-indexed coordinates (no duplicates).
* During evaluation, multiple `MWW.json` files will be used — your program must work with any valid file produced by `generate_mww.py`.

### Rewards and Termination

* **Step cost:** `-1` on every time step (including invalid moves).
* **Wumpus:** If the Wumpus appears, reward `-100` and episode terminates.
* **Goal:** Entering `G` gives reward `-1` and terminates the episode.

### Return

* Undiscounted return (`γ = 1`). Episode return is the sum of rewards until termination.
* Environment prints the cumulative reward at the end of each episode.

**Intuition:** The optimal behaviour is to discover the *shortest safe route* from `A` to `G` (maximizing return), balancing exploration against the high cost of encountering the Wumpus.

---

### Example `MWW.json`

```json
{
  "p": 0.20,
  "safe_path": [[1,1],[1,2],[1,3],[1,4],[2,4],[3,4],[4,4]]
}
```

> `p` is the Wumpus-appearance probability used for all non-safe cells. `safe_path` is the contiguous path from `A` to `G`.

---

## MagicWumpusWorld API (use only these functions)

Use only the following functions of `MagicWumpusWorld` in your code. Using any other functions or variables in `MagicWumpusWorld.py` will result in zero marks for that part.

* `CurrentState() -> (row, col) or None`
* `TakeAction(a: str) -> (reward: int, next_state: (row, col) or None)` where `a` ∈ {`"Up"`, `"Down"`, `"Left"`, `"Right"`}.
* `CumulativeRewardAndSteps() -> (reward, steps)`
* `reset(seed: Optional[int]) -> (1, 1)`

**Behavior summary:**

* Invalid moves keep the agent in place with reward `-1`.
* Entering `G` ends the episode with reward `-1`.
* Entering a non-safe cell triggers the Wumpus with probability `p` (if triggered: reward `-100` and termination).
* The environment prints a standardized step line for every action and prints the cumulative reward when the episode terminates.

---

## Assignment Tasks

### 1. Implement baseline TD-control algorithms

Implement and use the following TD-control algorithms for analysis and comparison in the report:

* **SARSA (on-policy):** `target = r_{t+1} + γ Q(s_{t+1}, a_{t+1})`.
* **Q-learning (off-policy):** `target = r_{t+1} + γ max_a Q(s_{t+1}, a)`.
* **Expected SARSA (on-policy expectation):** `target = r_{t+1} + γ Σ_a π(a|s_{t+1}) Q(s_{t+1}, a)`.

TD error: `δ = target − Q(s_t, a_t)` and update: `Q ← Q + α δ`.

> Use these baselines to generate the plots requested in the Report Generation section.

### 2. Compare action-selection strategies

Evaluate the performance of different exploration strategies, such as:

* ε-greedy (use both constant and decaying ε).
* Optimistic initial Q-values.
* UCB-style exploration (maintain state–action counts for confidence bonus).

### 3. Study hyperparameters

Vary and study the effects of:

* Discount factor `γ` and learning rate `α`.
* Any exploration parameters (ε schedule, UCB coefficient, optimistic init value).

---

## Plots and Evaluation

Run controlled experiments and include the following plots (well-labeled):

1. **Regret per episode** (lower is better):

   ```
   regret = -6 - (cumulative reward in the episode)
   ```

   * `-6` is the (hidden) return of the optimal shortest safe path. Regret should approach `0` as learning improves.

2. **Average TD-error magnitude per episode:**

   ```
   (1/T) Σ_{t=1..T} |δ_t|  for that episode
   ```

3. Any additional plots that help compare ideas (e.g., learning curves under different settings).

**Notes:** Use fixed random seeds when comparing methods and keep all other settings identical across runs.

---

## Requirements (code)

1. Implement **SARSA**, **Q-learning**, and **Expected SARSA** for analysis (report).
2. Implement an **improved TD algorithm** (this is the submission that will be run and graded). Design it to learn the safe path quickly and reliably.
3. **Program constraints:**

   * Use only the provided `MagicWumpusWorld` class.
   * Program must terminate within **45 seconds** on the evaluation machine (may stop earlier if converged).
   * When your program terminates it **must** print exactly the following two lines (replace placeholders):

```
Mean cummulative reward in the last 10 episodes :-XXX
Time taken by the algorithm : YY Seconds
```

---

## Report

* Explained the **improved TD algorithm** and justified the design choices (exploration, step-size schedule, initialization, etc.).
* Compare against the three baseline TD algorithms using the requested plots.
* Include hyperparameter settings and any ablation/negative results with short explanation.

---


# Improved Double-Q + UCB Tabular RL

This repository documents an **improved tabular temporal-difference (TD) algorithm** that combines Double-Q learning with UCB-style directed exploration, optimistic initialization, and per-(s,a) adaptive step-sizes. The method is designed for sparse, high-variance grid-world environments (e.g., Magic Wumpus World) and aims to reduce maximization bias, accelerate discovery of informative (s,a) pairs, and improve early returns and stability.

## Overview

Key components:

* **Double-Q (Q1, Q2)**: Two separate action-value tables to decouple action selection from value evaluation and suppress maximization bias.
* **UCB-augmented behaviour scores**: A visit-count dependent bonus added to the combined Q1 + Q2 value to guide directed exploration toward under-visited (s,a) pairs.
* **ε-greedy selection on augmented scores**: Adds an undirected stochastic element to avoid brittle deterministic behaviour; ε decays over episodes.
* **Optimistic initialization**: New (s,a) pairs start with high Q values to encourage first-time trials.
* **Per-(s,a) adaptive learning rates**: Step sizes decay with the number of visits to (s,a) but are clipped with a lower bound for stability.

These pieces are combined to produce an agent that explores intelligently early, avoids overestimation, and converges stably to near-optimal policies faster than conventional baselines.

## Algorithm (high-level)

1. Initialize two Q-tables `Q1(s,a) = Q2(s,a) = OPTIMISTIC_INIT` for all (s,a).

2. Keep a visit count table `N(s,a)` and per-(s,a) step sizes `alpha(s,a) = max(ALPHA_MIN, 1/(1 + N(s,a)))`.

3. At each decision state `s`, compute for every action `a`:

   ```text
   score(s,a) = Q1(s,a) + Q2(s,a) + bonus(s,a)
   bonus(s,a) = UCB_C * sqrt( log( total_visits + 1 ) / (1 + N(s,a)) )
   ```

   (The bonus formula above is a standard UCB-style term — any equivalent visit-dependent bonus may be used.)

4. Choose an action using `eps_greedy_over_scores(score(s,·))` with random tie-breaking probability to avoid deterministic cycles.

5. Observe transition `(s, a, r, s')`. Increment `N(s,a)`. Recompute `alpha(s,a)`.

6. With probability 0.5 pick one of the two Q-tables to update (Double-Q):

   * If updating `Q1`: let `a* = argmax_a Q1(s', a)`; target = `r + gamma * Q2(s', a*)` (or `r` when `s'` is terminal).
   * Update `Q1(s,a) <- Q1(s,a) + alpha(s,a) * (target - Q1(s,a))`.
   * Symmetric update for updating `Q2`.

7. Repeat until termination or early-stopping condition is met.

## Pseudocode

```python
# high-level pseudocode sketch
for episode in range(MAX_EPISODES):
    s = env.reset()
    for t in range(max_steps_per_episode):
        scores = {a: Q1[s,a] + Q2[s,a] + ucb_bonus(N, a) for a in actions}
        a = eps_greedy_over_scores(scores, eps=eps_schedule(episode))
        s2, r, done = env.step(a)
        N[s,a] += 1
        alpha = max(ALPHA_MIN, 1.0/(1 + N[s,a]))
        if random() < 0.5:
            a_star = argmax_a Q1[s2,a]
            target = r if done else r + gamma * Q2[s2,a_star]
            Q1[s,a] += alpha * (target - Q1[s,a])
        else:
            a_star = argmax_a Q2[s2,a]
            target = r if done else r + gamma * Q1[s2,a_star]
            Q2[s,a] += alpha * (target - Q2[s,a])
        if done: break
```

## Behaviour policy details

* **Combined behaviour score**: `S(s,a) = Q1(s,a) + Q2(s,a) + bonus(s,a)` used only for action selection, not for computing TD targets.
* **UCB bonus**: A function of `N(s,a)` and a hyperparameter `UCB_C` that prioritizes rarely visited actions early and decays as visits grow.
* **ε-greedy on scores**: Use ε-decay schedule `eps(t) = eps_end + (eps_start - eps_end) * exp(-(ep - 1)/tau)` to balance early stochastic exploration with later exploitation.
* **Random tie-breaking**: With a small probability of choosing uniformly among tied best scores to avoid repeat deterministic loops.

## Adaptive learning rate

Per-(s,a) step-size:

```
alpha(s,a) = max(ALPHA_MIN, 1 / (1 + N(s,a)))
```

This gives large early updates that decay as (s,a) get more data while ensuring learning never fully stalls (due to `ALPHA_MIN`).

## Hyperparameters (recommended defaults used in experiments)

* `GAMMA = 1.0` (episodic, shortest-path objectives)
* `EPS_START = 0.4`, `EPS_END = 0.02`, `EPS_DECAY_EPISODES = 800.0`
* `UCB_C = 1.5`
* `OPTIMISTIC_INIT = 5.0`
* `ALPHA_MIN = 0.02`
* `RANDOM_ACTION_PROB_ON_TIED = 0.2`
* `MAX_EPISODES = 3000` (or run until a time cutoff)
* `EARLY_STOP_TARGET = -6.0` (stop if mean return over last 10 episodes ≥ this)

These values were used in experiments on the Magic Wumpus World and are good starting points for similar grid environments.

## Why this works (intuitions)

* **Double-Q** reduces maximization bias by separating argmax selection and value evaluation across two tables, producing more conservative and accurate targets.
* **UCB bonus + optimistic init** provides directed exploration toward uncertain (s,a) pairs and ensures fast first-pass coverage of the state-action space.
* **ε-greedy smoothing** prevents brittle deterministic behavior during early exploration.
* **Per-(s,a) adaptive α** stabilizes learning by reducing step-sizes as local counts grow while guaranteeing continued adaptability.

## Comparison to baselines

* **vs Q-learning**: Double-Q avoids overestimation and large transient spikes; UCB and optimistic init speed up discovery of good actions.
* **vs SARSA**: Directed UCB exploration finds informative actions faster than SARSA’s undirected ε-greedy exploration, giving better early returns.
* **vs Expected SARSA**: Expected updates are stable, but still lack directed exploration; adding UCB accelerates improvement while Double-Q avoids overoptimistic traps.

## Expected empirical patterns

* Faster early learning and steeper improvement in returns.
* Lower initial TD-error magnitudes and quicker decay of TD errors.
* Reduced early regret and fewer large oscillations compared to Q-learning.
* Asymptotic performance comparable to or slightly better than baselines when hyperparameters are well tuned.

## Experimental setup (Magic Wumpus World)

* Environment: `MagicWumpusWorld()` (same map for all methods)
* Seeds: `0..9` and report mean ± std across seeds
* Episodes: `MAX_EPISODES = 3000` (or a fixed time cutoff, e.g., 44.9s)
* Discount: `GAMMA = 1.0`
* Metrics recorded per episode: `return`, `avg_td_error`, `N_sa` statistics, runtime
* Baselines: SARSA, Q-learning, Expected SARSA (identical ε and α schedules for fairness)

## Results summary

Brief findings from experiments:

* All algorithms show high initial regret due to exploration.
* Q-learning exhibited the largest early regret and slowest decline.
* SARSA and Expected SARSA reduced regret faster than Q-learning but were noisier.
* **Improved algorithm** (Double-Q + UCB + optimistic init + adaptive α) achieved the lowest regret and reached near-zero regret significantly earlier.
* Returns and TD errors follow similar patterns: improved algorithm climbs to near-optimal returns faster and shows smaller TD-error magnitudes.

> Experimental result files: `MMW.json`, `MMW_15.json`, `MMW_25.json`, `MMW_35.json`, `MMW_45.json` (example result dumps used in evaluation).

## Practical tuning notes

* Increase `UCB_C` to encourage stronger directed exploration but expect higher variance.
* Large `OPTIMISTIC_INIT` accelerates first-visit exploration but can inflate early regret and variance in noisy settings.
* `ALPHA_MIN` prevents vanishing learning; set conservatively (e.g., 0.02–0.05).
* Choose `gamma` to reflect task horizon: for shortest-path grid tasks `gamma = 1.0` is reasonable; use lower gamma for tasks prioritizing immediate reward.

## Usage

1. Implement or import a tabular environment (states and discrete actions). The Magic Wumpus World used in the experiments is one example.
2. Initialize `Q1`, `Q2`, and `N` arrays with appropriate shapes.
3. Use the pseudocode above as an implementation sketch and tune hyperparameters for your environment.

---
