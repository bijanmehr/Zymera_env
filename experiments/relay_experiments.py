"""Relay experiment set (CPU, Regime B). Three experiments, each ES-trained on TRAIN scales and
evaluated on EVAL scales (the larger one is HELD-OUT = generalization). Reports the four metrics:
delivered coverage, connectivity, mission-safety (degree-floor violation rate), relay fraction.

  baseline  : fixed w_conn=1.0, standard episode length         (reference)
  adaptive  : ADAPTIVE w_conn — Lagrangian λ self-tunes each gen toward a connectivity target
  fiedler   : role head also sees λ₂ + own Fiedler component    (connectivity signal)
  episodes  : fixed w_conn=1.0 but ~2x episode length           (more steps to sweep)

  PYTHONPATH=. .venv/bin/python -m experiments.relay_experiments
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
from swarm_explore.relay_mission import (rollout_mission, make_theta_policy, make_fiedler_policy,
                                         init_theta_n)
from swarm_explore.core import random_wall
from experiments._relay_exp_worker import eval_one

OUT = os.path.normpath(os.path.join(os.path.dirname(__file__), "runs", "relay-experiments"))
os.makedirs(OUT, exist_ok=True)

TRAIN = [(16, 4), (24, 6)]
EVAL = [(16, 4), (24, 6), (32, 10)]          # 32² is HELD-OUT (not trained) -> generalization
STD = {16: 110, 24: 170, 32: 240}
LONG = {16: 220, 24: 340, 32: 480}
GENS, POP, SEEDS, CONN_TARGET = int(os.environ.get("REXP_GENS", "20")), 8, 2, 0.9


def es_train(mode, steps, adaptive, pool, rng):
    theta = (init_theta_n(9) if mode == "fiedler" else arch3.init_theta(0)).astype(np.float64)
    w_conn = 1.0; hist = []
    for g in range(GENS):
        eps = rng.normal(0, 1, (POP, theta.size)); eps = np.concatenate([eps, -eps], 0)
        units = [(k, theta + 0.15 * eps[k], mode, grid, n, 5, steps[grid], s, w_conn)
                 for k in range(len(eps)) for (grid, n) in TRAIN for s in range(SEEDS)]
        acc = collections.defaultdict(list)
        for r in pool.map(eval_one, units, chunksize=2):
            acc[r[0]].append(r[1])
        F = np.array([float(np.mean(acc[k])) for k in range(len(eps))])
        A = (F - F.mean()) / (F.std() + 1e-8)
        theta = theta + 0.08 * (eps.T @ A) / (len(eps) * 0.15)
        if adaptive:                                            # Lagrangian: drive connectivity to target
            cur = list(pool.map(eval_one, [("c", theta, mode, grid, n, 5, steps[grid], s, w_conn)
                                           for (grid, n) in TRAIN for s in range(SEEDS)]))
            conn = float(np.mean([r[3] for r in cur]))
            w_conn = float(np.clip(w_conn + 0.5 * (CONN_TARGET - conn), 0.1, 5.0))
            hist.append(round(w_conn, 2))
    return theta, w_conn, hist


def evaluate(theta, mode, steps):
    out = []
    for (grid, n) in EVAL:
        K = max(1, n - 2)
        pol = make_fiedler_policy(theta, K) if mode == "fiedler" else make_theta_policy(theta, K)
        acc = np.zeros(4)
        for s in range(3):
            m = rollout_mission(grid, grid, random_wall(grid, grid, 0.05, 300 + s), n, 5, 3,
                                steps[grid], 300 + s, pol, backstop=False, decay=0.97)
            acc += [m["delivered_cov"], m["connectivity"], m["safety_viol"], m["relay_frac"]]
        acc /= 3
        out.append(dict(grid=grid, n=n, heldout=(grid, n) not in TRAIN,
                        cov=round(acc[0], 3), conn=round(acc[1], 3),
                        safety=round(acc[2], 3), relay=round(acc[3], 3)))
    return out


def main():
    workers = min(mp.cpu_count(), 24)
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    rng = np.random.default_rng(0)
    log = open(os.path.join(OUT, "relay_experiments.log"), "w")

    def say(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()

    experiments = [
        ("baseline", "standard", STD, False),
        ("adaptive", "standard", STD, True),
        ("fiedler", "fiedler", STD, False),
        ("episodes", "standard", LONG, False),
    ]
    results = {}
    say(f"Relay experiment set on {workers} cores · train {TRAIN} · eval {EVAL} (32²=held-out)")
    try:
        for name, mode, steps, adaptive in experiments:
            theta, w_final, whist = es_train(mode, steps, adaptive, pool, rng)
            per = evaluate(theta, mode, steps)
            results[name] = dict(mode=mode, w_conn_final=round(w_final, 2), w_hist=whist, per_scale=per)
            say(f"[{name}] w_conn_final={w_final:.2f}")
            for p in per:
                tag = " (held-out)" if p["heldout"] else ""
                say(f"   {p['grid']}²/N{p['n']}{tag}: cov {p['cov']*100:3.0f}%  conn {p['conn']*100:3.0f}%  "
                    f"safety_viol {p['safety']:.3f}  relay {p['relay']*100:3.0f}%")
            json.dump(results, open(os.path.join(OUT, "experiments.json"), "w"), indent=2)
    finally:
        pool.shutdown()
    say("=== done ===")


if __name__ == "__main__":
    main()
