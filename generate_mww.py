#!/usr/bin/env python3
# generate_mww.py
# Create ./MWW.json with a random p and a random safe (shortest) path from (1,1) to (4,4)

import json
import random
from typing import List, Tuple

ALLOWED_P = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

def random_safe_path_shortest() -> List[Tuple[int, int]]:
    """
    Generate a random shortest path from (1,1) to (4,4) on a 4x4 grid.
    With (1,1) at bottom-left, a shortest path is any permutation of
    3 'Up' moves and 3 'Right' moves.
    """
    r, c = 1, 1
    path = [(r, c)]
    moves = ['Up'] * 3 + ['Right'] * 3
    random.shuffle(moves)
    for m in moves:
        if m == 'Up':
            r += 1
        else:  # 'Right'
            c += 1
        path.append((r, c))
    # Should end at (4,4)
    assert path[0] == (1, 1) and path[-1] == (4, 4), "Path generation error."
    return path

def main() -> None:
    p = random.choice(ALLOWED_P)  # uniform over allowed values
    safe_path = random_safe_path_shortest()

    data = {
        "p": p,
        "safe_path": [[r, c] for (r, c) in safe_path]  # as 1-indexed [row, col]
    }

    with open("MWW.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Created MWW.json with p={p:.2f} and safe_path of length {len(safe_path)-1}.")

if __name__ == "__main__":
    main()
