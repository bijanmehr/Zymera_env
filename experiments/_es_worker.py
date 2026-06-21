"""CPU worker: evaluate ONE controller theta on ONE (grid, N, seed) -> fitness + metrics.

Pinned to CPU so the ES population (and the held-out eval) fan out across cores
without contending for the GPU. The metrics arch3 returns are all normalized
fractions, so the fitness is directly comparable across scales (size-invariant).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np


def eval_one(args):
    tid, theta, grid, n, comm_r, steps, seed, K, w_conn, cost_relay = args
    from swarm_explore import arch3
    from swarm_explore.core import random_wall
    wall = random_wall(grid, grid, 0.05, seed)
    m = arch3.run_arch(grid, grid, wall, n, comm_r, 3, steps, seed, np.asarray(theta),
                       K=K, lam_frontier=1.0)
    fit = (m["coverage"] + w_conn * m["largest_comp_frac"]
           - cost_relay * m["relay_frac"] - 0.2 * m["safety_viol"])
    return [tid, float(fit), float(m["coverage"]), float(m["connectivity"]),
            float(m["largest_comp_frac"]), float(m["relay_frac"]), float(m["safety_viol"])]
