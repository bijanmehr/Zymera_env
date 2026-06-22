"""Standalone CPU parameter sweep for the relay-mission's Regime B (soft + latency-discounted
delivery), which collapsed to dispersal under a FIXED w_conn=0.5. Question: does raising the
connectivity weight (and tuning the latency decay) make the soft penalty BIND — i.e. recover
connectivity without the full Lagrangian? ES-trains one controller per (w_conn, decay), evals
per scale. Pinned to CPU so it runs on the cores alongside a GPU job.

  PYTHONPATH=. .venv/bin/python -m experiments.relay_param_sweep
"""
import os
import sys
import json
import collections
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from swarm_explore import arch3
from swarm_explore.relay_mission import rollout_mission, make_theta_policy
from swarm_explore.core import random_wall
from experiments._relay_es_worker import eval_one

OUT = os.path.normpath(os.path.join(os.path.dirname(__file__), "runs", "relay-psweep"))
os.makedirs(OUT, exist_ok=True)

W_CONNS = [float(x) for x in os.environ.get("PSWEEP_WCONN", "0.5,1.0,2.0,4.0,8.0").split(",")]
DECAYS = [float(x) for x in os.environ.get("PSWEEP_DECAY", "0.9,0.97").split(",")]
SCALES = [(16, 4), (24, 6)]
STEPS = {16: 110, 24: 170}
GENS, POP, SEEDS = int(os.environ.get("PSWEEP_GENS", "18")), 8, 2


def train_es(w_conn, decay, pool, rng):
    theta = arch3.init_theta(0).astype(np.float64)
    for g in range(GENS):
        eps = rng.normal(0, 1, (POP, theta.size)); eps = np.concatenate([eps, -eps], 0)
        units = [(k, theta + 0.15 * eps[k], "B", grid, n, 5, STEPS[grid], s, w_conn, decay)
                 for k in range(len(eps)) for (grid, n) in SCALES for s in range(SEEDS)]
        acc = collections.defaultdict(list)
        for r in pool.map(eval_one, units, chunksize=2):
            acc[r[0]].append(r[1])
        F = np.array([float(np.mean(acc[k])) for k in range(len(eps))])
        A = (F - F.mean()) / (F.std() + 1e-8)
        theta = theta + 0.08 * (eps.T @ A) / (len(eps) * 0.15)
    return theta


def main():
    workers = min(mp.cpu_count(), 24)
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    rng = np.random.default_rng(0)
    results = []
    log = open(os.path.join(OUT, "psweep.log"), "w")

    def say(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()

    say(f"Regime-B param sweep: {len(W_CONNS)}x{len(DECAYS)} (w_conn x decay) on {workers} cores")
    try:
        for decay in DECAYS:
            for w_conn in W_CONNS:
                theta = train_es(w_conn, decay, pool, rng)
                row = dict(w_conn=w_conn, decay=decay, per_scale=[])
                for (grid, n) in SCALES:
                    acc = np.zeros(4)
                    for s in range(3):
                        m = rollout_mission(grid, grid, random_wall(grid, grid, 0.05, 200 + s), n, 5, 3,
                                            STEPS[grid], 200 + s, make_theta_policy(theta, n - 2),
                                            backstop=False, decay=decay)
                        acc += [m["delivered_cov"], m["raw_cov"], m["connectivity"], m["relay_frac"]]
                    acc /= 3
                    row["per_scale"].append(dict(grid=grid, n=n, cov=round(acc[0], 3), raw=round(acc[1], 3),
                                                 conn=round(acc[2], 3), relay=round(acc[3], 3)))
                results.append(row)
                big = row["per_scale"][-1]
                say(f"  w_conn={w_conn:<4} decay={decay}: "
                    + " | ".join(f"{p['grid']}² cov {p['cov']*100:.0f}% conn {p['conn']*100:.0f}% relay {p['relay']*100:.0f}%"
                                 for p in row["per_scale"]))
                json.dump(results, open(os.path.join(OUT, "sweep.json"), "w"), indent=2)
    finally:
        pool.shutdown()
    say("=== done ===")


if __name__ == "__main__":
    main()
