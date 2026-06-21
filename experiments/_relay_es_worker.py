"""CPU worker: evaluate ONE relay-mission controller theta on ONE (regime, scale, seed).

Regime A (hard backstop): connectivity is guaranteed, so fitness = delivered coverage.
Regime B (soft, latency-discounted delivery): fitness = delivered_score + w_conn * connectivity,
so the learner must buy connectivity back through good relaying.
Pinned to CPU so the ES population fans out across cores without GPU contention.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np


def eval_one(args):
    tid, theta, regime, grid, n, comm_r, steps, seed, w_conn, decay = args
    from swarm_explore.relay_mission import rollout_mission, make_theta_policy
    from swarm_explore.core import random_wall
    hard = (regime == "A")
    wall = random_wall(grid, grid, 0.05, seed)
    m = rollout_mission(grid, grid, wall, n, comm_r, 3, steps, seed,
                        make_theta_policy(np.asarray(theta), n - 2),
                        backstop=hard, decay=(1.0 if hard else decay))
    if hard:
        fit = m["delivered_cov"] - 0.2 * m["safety_viol"]
    else:
        fit = m["delivered_score"] + w_conn * m["connectivity"] - 0.2 * m["safety_viol"]
    spread = float(np.std(m["per_agent_relay"]))                      # role specialization signal
    return [tid, float(fit), float(m["delivered_cov"]), float(m["raw_cov"]),
            float(m["connectivity"]), float(m["relay_frac"]), spread]
