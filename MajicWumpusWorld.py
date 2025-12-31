#!/usr/bin/env python3
# MajicWumpusWorld.py
"""
Magic Wumpus World (4x4) — reads ./MWW.json automatically (no CLI path)
"""

from __future__ import annotations
import json
import os
import random
import sys
import argparse
from typing import List, Tuple, Optional, Set

Coord = Tuple[int, int]


class MagicWumpusWorld:
    ROWS: int = 4
    COLS: int = 4
    START: Coord = (1, 1)  # bottom-left = 'A'
    GOAL: Coord = (4, 4)   # top-right  = 'G'
    ACTIONS = ("Up", "Down", "Left", "Right")

    def __init__(self, seed: Optional[int] = None):
        json_path = os.path.join(os.getcwd(), "MWW.json")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(
                f"MWW.json not found in current folder: {os.getcwd()}\n"
                f"Expected a file named 'MWW.json' with keys 'p' and 'safe_path'."
            )

        with open(json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        if "p" not in cfg or not isinstance(cfg["p"], (int, float)):
            raise ValueError("MWW.json must contain numeric key 'p' in [0,1].")
        self.p: float = float(cfg["p"])
        if not (0.0 <= self.p <= 1.0):
            raise ValueError(f"'p' must be in [0,1], got {self.p}.")

        if "safe_path" not in cfg or not isinstance(cfg["safe_path"], list):
            raise ValueError("MWW.json must contain key 'safe_path' as a list of coordinates.")
        self.safe_path: List[Coord] = self._normalize_path(cfg["safe_path"])
        self.safe_set: Set[Coord] = set(self.safe_path)
        self._validate_safe_path()

        self._rng = random.Random(seed)
        self._state: Optional[Coord] = self.START  # None means terminal

        # Episode cumulative reward (gamma = 1)
        self._episode_return: int = 0

        # --- ADDED: per-episode step counter ---
        self._step: int = 0

    # ---------- Public API ----------

    def CurrentState(self) -> Optional[Coord]:
        return self._state

    def CumulativeRewardAndSteps(self) -> int:
        return self._episode_return, self._step

    def TakeAction(self, a: str) -> Tuple[int, Optional[Coord]]:
        """
        Apply action a. Prints the standardized step line every call, and
        prints the termination message AFTER that line if the episode ends.
        """
        if self._state is None:
            raise RuntimeError("Episode already ended. Call reset() to start a new episode.")

        a = self._normalize_action(a)
        r, c = self._state
        dr, dc = self._delta(a)
        nr, nc = r + dr, c + dc

        # Increment step counter once per attempted action
        self._step += 1
        step = self._step

        # Invalid move: stay and pay -1
        if not self._in_bounds(nr, nc):
            reward = -1
            next_state: Optional[Coord] = self._state
            self._episode_return += reward
            # Print the required step line
            print(f"Step {step:03d}: action={a:>5} | reward={reward:>4} | next_state={next_state}")
            return reward, next_state

        next_cell = (nr, nc)

        # Goal reached: end with -1; print step line first, then the goal message last
        if next_cell == self.GOAL:
            reward = -1
            next_state = None
            print(f"Step {step:03d}: action={a:>5} | reward={reward:>4} | next_state={next_state}")
            return self._terminate_episode(reward, reason="“Target reached — well done!”")

        # Hazard sampling only on non-safe cells
        if next_cell not in self.safe_set and self._rng.random() < self.p:
            reward = -100
            next_state = None
            print(f"Step {step:03d}: action={a:>5} | reward={reward:>4} | next_state={next_state}")
            return self._terminate_episode(reward, reason="“Wumpus attack — it’s over!”")

        # Safe (or survived hazard): move and pay step cost
        reward = -1
        self._state = next_cell
        self._episode_return += reward
        next_state = self._state
        print(f"Step {step:03d}: action={a:>5} | reward={reward:>4} | next_state={next_state}")
        return reward, next_state

    def reset(self, seed: Optional[int] = None) -> Coord:
        if seed is not None:
            self._rng.seed(seed)
        self._state = self.START
        self._episode_return = 0
        self._step = 0  # reset step counter
        return self._state

    # ---------- Internal helpers ----------

    def _terminate_episode(self, reward: int, reason: str) -> Tuple[int, None]:
        self._episode_return += reward
        self._state = None
        creward,steps = self.CumulativeRewardAndSteps()
        # Print the requested termination message as the last line
        print(f"{reason} → Cummulative reward = {creward}, Total time steps = {steps}")
        return reward, None

    @staticmethod
    def _normalize_action(a: str) -> str:
        if not isinstance(a, str):
            raise ValueError("Action must be a string.")
        a = a.strip()
        for name in MagicWumpusWorld.ACTIONS:
            if a.lower() == name.lower():
                return name
        raise ValueError(f"Invalid action '{a}'. Valid actions: {MagicWumpusWorld.ACTIONS}")

    @staticmethod
    def _delta(a: str) -> Tuple[int, int]:
        return {
            "Up":    (1, 0),
            "Down":  (-1, 0),
            "Left":  (0, -1),
            "Right": (0, 1),
        }[a]

    @classmethod
    def _in_bounds(cls, r: int, c: int) -> bool:
        return 1 <= r <= cls.ROWS and 1 <= c <= cls.COLS

    @staticmethod
    def _normalize_path(raw_path: List) -> List[Coord]:
        norm: List[Coord] = []
        for item in raw_path:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                r, c = int(item[0]), int(item[1])
            elif isinstance(item, dict) and {"row", "col"} <= set(item.keys()):
                r, c = int(item["row"]), int(item["col"])
            else:
                raise ValueError("Each safe_path element must be [row,col] or {'row':..,'col':..}.")
            norm.append((r, c))
        return norm

    def _validate_safe_path(self) -> None:
        if self.safe_path[0] != self.START:
            raise ValueError(f"safe_path must start at {self.START}, got {self.safe_path[0]}.")
        if self.safe_path[-1] != self.GOAL:
            raise ValueError(f"safe_path must end at {self.GOAL}, got {self.safe_path[-1]}.")
        if len(set(self.safe_path)) != len(self.safe_path):
            raise ValueError("safe_path contains duplicate cells.")
        for (r, c) in self.safe_path:
            if not self._in_bounds(r, c):
                raise ValueError(f"safe_path contains out-of-bounds cell {(r, c)}.")
        for (r1, c1), (r2, c2) in zip(self.safe_path, self.safe_path[1:]):
            if abs(r1 - r2) + abs(c1 - c2) != 1:
                raise ValueError(f"safe_path must move one cell at a time (got {(r1,c1)} -> {(r2,c2)}).")

    @classmethod
    def cell_name(cls, cell: Coord) -> str:
        if cell == cls.START:
            return "A"
        if cell == cls.GOAL:
            return "G"
        r, c = cell
        return f"({r},{c})"


def _interactive_loop(env: MagicWumpusWorld) -> None:
    print("Interactive Magic Wumpus World (Up/Down/Left/Right; 'q' quit, 'r' reset)")
    print(f"Start at {env.cell_name(env.CurrentState())} = {env.CurrentState()}; Goal is G={env.GOAL}")
    while True:
        s = env.CurrentState()
        if s is None:
            print(f"Episode over. Cumulative reward was {env.CumulativeReward()}.")
            cmd = input("Type 'r' to reset or 'q' to quit > ").strip()
            if cmd.lower() == "r":
                env.reset()
                print(f"Reset to {env.CurrentState()} (A).")
                continue
            break

        cmd = input("Action > ").strip()
        if cmd.lower() == "q":
            break
        try:
            reward, next_state = env.TakeAction(cmd)
        except Exception as e:
            print(f"Error: {e}")
            continue
        # (No change here; keeping the example output as-is)
        print(f"  -> reward={reward}, next_state={next_state}")
        if next_state is None:
            print("  Episode terminated.")


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")
    ap.add_argument("--interactive", action="store_true", help="Run a tiny interactive demo.")
    args = ap.parse_args(argv)

    env = MagicWumpusWorld(seed=args.seed)
    print(f"Loaded world from ./MWW.json: p={env.p}, safe_path length={len(env.safe_path)}")
    print(f"Start A={env.START}, Goal G={env.GOAL}")

    if args.interactive:
        _interactive_loop(env)
    else:
        sequence = ["Right", "Right", "Right", "Up", "Up", "Up"]
        for a in sequence:
            s = env.CurrentState()
            if s is None:
                print(f"(Episode ended. Cumulative reward was {env.CumulativeReward()})")
                break
            reward, ns = env.TakeAction(a)
            print(f"  {a:>5} | reward={reward:>4} | next_state={ns}")
        if env.CurrentState() is not None:
            print("(Demo finished; episode may still be running.)")
            print(f"Current cumulative reward: {env.CumulativeReward()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
