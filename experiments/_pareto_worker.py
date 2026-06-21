"""CPU-only worker for the pareto / wall-boundary sweep.

Runs the numpy swarm engine (arch3.run_arch). Forced onto CPU so a poolful of
workers does not contend for the GPU (the engine uses jax only for tiny env steps);
this leaves the GPU free for the belief job.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def run_one(args):
    lam, grid, n, comm_r, seed, steps = args
    from swarm_explore import arch3
    from swarm_explore.core import random_wall
    theta = arch3.init_theta(0).copy()
    theta[-1] = -10.0                       # compass-only baseline (P(relay) ~ 0)
    wall = random_wall(grid, grid, 0.05, seed)
    m = arch3.run_arch(grid, grid, wall, n, comm_r, 3, steps, seed, theta,
                       K=max(1, n - 1 - 2), lam_frontier=lam)
    return [lam, grid, n, comm_r, float(m["coverage"]), float(m["connectivity"]), float(m["largest_comp_frac"])]
