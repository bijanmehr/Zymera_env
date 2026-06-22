"""CPU worker for the relay experiment set (adaptive / fiedler / more-episodes). Evaluates one
theta on one (mode, grid, N, seed) in Regime B (soft, latency-discounted), returning fitness +
the four metrics we track: delivered coverage, connectivity, relay fraction, mission-safety.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np


def eval_one(args):
    tid, theta, mode, grid, n, comm_r, steps, seed, w_conn = args
    from swarm_explore.relay_mission import rollout_mission, make_theta_policy, make_fiedler_policy
    from swarm_explore.core import random_wall
    K = max(1, n - 2)
    pol = (make_fiedler_policy(np.asarray(theta), K) if mode == "fiedler"
           else make_theta_policy(np.asarray(theta), K))
    m = rollout_mission(grid, grid, random_wall(grid, grid, 0.05, seed), n, comm_r, 3, steps, seed,
                        pol, backstop=False, decay=0.97)
    fit = m["delivered_score"] + w_conn * m["connectivity"] - 0.2 * m["safety_viol"]
    return [tid, float(fit), float(m["delivered_cov"]), float(m["connectivity"]),
            float(m["relay_frac"]), float(m["safety_viol"])]
