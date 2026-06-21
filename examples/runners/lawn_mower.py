"""
Lawn-mower coverage — the simplest reactive baseline.

One agent, local sensing only (its own position + which neighbours are
blocked). The algorithm, in three moves:

  1. SEEK  — go NORTH until blocked, then WEST until blocked: reach the
             upper-left corner.
  2. SWEEP — mow rows: run along the heading (EAST/WEST); at the lane end
             drop SOUTH one row and flip direction (serpentine).
  3. CIRCLE — hit an obstacle? go around it CLOCKWISE (prefer right turns)
             until the lane can continue.

It is deliberately NOT complete: circling an obstacle and rejoining the
lane skips cells stranded in pockets behind it. That's the price of a
purely reactive agent with no map. Coverage is reported honestly.

Usage::

    python examples/lawn_mower.py --grid 12 --obstacles 8 --output lawn.gif
"""

from __future__ import annotations

import argparse
from collections import deque
from typing import Dict, List, Set, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import zymera

Cell = Tuple[int, int]

# ---- directions ------------------------------------------------------------
# ActionId values are already clockwise: NORTH=1 → EAST=2 → SOUTH=3 → WEST=4.
A = zymera.ActionId
_DELTA: Dict[int, Cell] = {
    int(A.STAY): (0, 0), int(A.NORTH): (-1, 0), int(A.EAST): (0, 1),
    int(A.SOUTH): (1, 0), int(A.WEST): (0, -1),
}
NORTH, EAST, SOUTH, WEST, STAY = (int(A.NORTH), int(A.EAST), int(A.SOUTH),
                                  int(A.WEST), int(A.STAY))


def _add(p: Cell, d: int) -> Cell:
    dr, dc = _DELTA[d]
    return (p[0] + dr, p[1] + dc)


def _right(d: int) -> int:
    """One 90° clockwise (right) turn: N→E→S→W→N."""
    return d % 4 + 1


def _clockwise_from(heading: int) -> List[int]:
    """The four directions in clockwise order, starting to the RIGHT of
    ``heading`` — the turn-preference for circling an obstacle."""
    out, d = [], _right(heading)
    for _ in range(4):
        out.append(d)
        d = _right(d)
    return out


# ============================================================================
# The agent — local-only. Memory: a visited set, a heading, a phase.
# ============================================================================


class LawnMower:
    SEEK, SWEEP = 0, 1

    def __init__(self) -> None:
        self.visited: Set[Cell] = set()
        self.phase = self.SEEK
        self.heading = EAST

    def act(self, pos: Cell, readings: Dict[Cell, bool]) -> int:
        self.visited.add(pos)

        def free(d: int) -> bool:                     # neighbour sensed free?
            return not readings.get(_add(pos, d), True)   # unknown / wall / edge → blocked
        def fresh(d: int) -> bool:                    # free AND not yet visited
            return free(d) and _add(pos, d) not in self.visited

        # 1. SEEK the upper-left corner: up, then left.
        if self.phase == self.SEEK:
            if fresh(NORTH): return NORTH
            if fresh(WEST):  return WEST
            self.phase = self.SWEEP
            self.heading = EAST

        # 2. SWEEP: run the lane.
        if fresh(self.heading):
            return self.heading
        # 3. lane end / blocked ahead: drop a row, flip direction.
        if fresh(SOUTH):
            self.heading = WEST if self.heading == EAST else EAST
            return SOUTH
        # 4. circle the obstacle clockwise onto fresh ground (prefer right).
        for d in _clockwise_from(self.heading):
            if fresh(d):
                return d
        # 5. dead end: escape clockwise over already-visited cells.
        for d in _clockwise_from(self.heading):
            if free(d):
                return d
        return STAY                                   # fully boxed in


# ============================================================================
# Harness — obstacles, reachability accounting, the run loop
# ============================================================================


def make_obstacles(rng: np.random.Generator, grid_h: int, grid_w: int,
                   n_obstacles: int) -> Set[Cell]:
    cells = [(r, c) for r in range(grid_h) for c in range(grid_w)]
    n = min(n_obstacles, len(cells))
    return {cells[i] for i in rng.choice(len(cells), size=n, replace=False)}


def obstacle_mask(obstacles: Set[Cell], grid_h: int, grid_w: int) -> np.ndarray:
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    for r, c in obstacles:
        mask[r, c] = True
    return mask


def reachable_free_cells(spawn: Cell, grid_h: int, grid_w: int,
                         obstacles: Set[Cell]) -> Set[Cell]:
    """4-connected free cells reachable from spawn (reporting only)."""
    seen, q = {spawn}, deque([spawn])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (r + dr, c + dc)
            if (0 <= nb[0] < grid_h and 0 <= nb[1] < grid_w
                    and nb not in obstacles and nb not in seen):
                seen.add(nb)
                q.append(nb)
    return seen


def run_coverage(env, agent: LawnMower, obstacles: Set[Cell], key,
                 patience: int = 64, hard_cap: int = 5000):
    """Drive the env until the agent quits, stops making progress, or hits a
    cap. Returns ``(frames, n_steps, stop_reason)``."""
    _, state = env.reset(key)
    frames = [state]
    stale = 0
    while True:
        pos: Cell = tuple(int(x) for x in state.body.position[0])  # type: ignore[assignment]
        before = len(agent.visited)
        action_id = agent.act(pos, env.sensor.observe_one(state, 0))
        if action_id == STAY:
            return frames, len(frames) - 1, "boxed-in"
        assert _add(pos, action_id) not in obstacles, "agent targeted an obstacle!"
        _, state, _, _, _ = env.step(state, jnp.array([action_id], jnp.int32), key)
        frames.append(state)
        stale = 0 if len(agent.visited) > before else stale + 1
        if stale >= patience:
            return frames, len(frames) - 1, "no-progress"
        if len(frames) - 1 >= hard_cap:
            return frames, len(frames) - 1, "hard-cap"


# ---- rendering -------------------------------------------------------------


def draw_obstacles(ax, obstacles: Set[Cell]) -> None:
    if obstacles:
        rows, cols = zip(*obstacles)
        ax.scatter(cols, rows, c="dimgray", s=380, marker="s",
                   edgecolors="black", linewidths=0.8, zorder=2)


def write_gif(frames, obstacles, grid_h, grid_w, output, fps):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from zymera.viz import draw_frame, draw_grid_lines

    fig, ax = plt.subplots(figsize=(5, 5))

    def update(t):
        draw_frame(ax, frames[t], title=f"step {t}")
        draw_obstacles(ax, obstacles)
        draw_grid_lines(ax, grid_h, grid_w)
        return ()

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps)
    anim.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)


# ---- main ------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Reactive lawn-mower coverage.")
    ap.add_argument("--grid",      type=int, default=12)
    ap.add_argument("--obstacles", type=int, default=8)
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--output",    type=str, default="lawn.gif")
    ap.add_argument("--fps",       type=int, default=10)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)

    obstacles = make_obstacles(rng, args.grid, args.grid, args.obstacles)
    wall = obstacle_mask(obstacles, args.grid, args.grid)
    env = zymera.make("empty-v0", grid_h=args.grid, grid_w=args.grid, n_agents=1,
                      wall=wall, sense_radius=1)
    _, state = env.reset(key)
    spawn: Cell = tuple(int(x) for x in state.body.position[0])  # type: ignore[assignment]

    frames, n_steps, why = run_coverage(env, LawnMower(), obstacles, key)

    visited_mask = np.asarray(frames[-1].explored) > 0
    reachable = reachable_free_cells(spawn, args.grid, args.grid, obstacles)
    covered = sum(1 for cell in reachable if visited_mask[cell])
    write_gif(frames, obstacles, args.grid, args.grid, args.output, args.fps)

    print(f"wrote {args.output}  "
          f"({n_steps} steps, {covered}/{len(reachable)} reachable cells covered, "
          f"stopped: {why}, {args.grid}×{args.grid}, {len(obstacles)} obstacles, seed {args.seed})")


if __name__ == "__main__":
    main()
