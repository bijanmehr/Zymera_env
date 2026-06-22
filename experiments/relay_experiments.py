"""Relay experiment set (CPU, Regime B) as a full GENERALIZATION study:
  TRAIN     on a range of worlds INCLUDING big ones  (16² + 24² + 32², domain-randomized)
  ZERO-SHOT to held-out sizes with no further training (20² interp · 40² extrapolation)
  FEW-SHOT  adapt the trained policy a few generations on a held-out size (40²) — the adaptation curve

Four experiments, each reporting coverage · connectivity · mission-safety (degree-floor violations) · relay:
  baseline : fixed w_conn=1.0, standard episodes        adaptive : Lagrangian λ self-tunes to a conn target
  fiedler  : role head sees λ₂ + own Fiedler component  episodes : ~2x episode length

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

TRAIN = [(16, 4), (24, 6), (32, 10)]         # train INCLUDING big worlds
ZEROSHOT = [(20, 5), (40, 16)]               # held-out: 20² interpolation, 40² extrapolation
FEWSHOT = (40, 16)                           # few-shot adapt here
STD = {16: 110, 20: 140, 24: 170, 32: 240, 40: 320}
LONG = {k: v * 2 for k, v in STD.items()}
GENS, POP, SEEDS, CONN_TARGET = int(os.environ.get("REXP_GENS", "20")), 8, 2, 0.9
FEWSHOT_GENS = int(os.environ.get("REXP_FEWGENS", "6"))


def es_train(mode, steps, adaptive, scales, gens, pool, rng, theta0=None):
    theta = (theta0.copy() if theta0 is not None
             else (init_theta_n(9) if mode == "fiedler" else arch3.init_theta(0)).astype(np.float64))
    w_conn = 1.0
    for g in range(gens):
        eps = rng.normal(0, 1, (POP, theta.size)); eps = np.concatenate([eps, -eps], 0)
        units = [(k, theta + 0.15 * eps[k], mode, grid, n, 5, steps[grid], s, w_conn)
                 for k in range(len(eps)) for (grid, n) in scales for s in range(SEEDS)]
        acc = collections.defaultdict(list)
        for r in pool.map(eval_one, units, chunksize=2):
            acc[r[0]].append(r[1])
        F = np.array([float(np.mean(acc[k])) for k in range(len(eps))])
        A = (F - F.mean()) / (F.std() + 1e-8)
        theta = theta + 0.08 * (eps.T @ A) / (len(eps) * 0.15)
        if adaptive:
            cur = list(pool.map(eval_one, [("c", theta, mode, grid, n, 5, steps[grid], s, w_conn)
                                           for (grid, n) in scales for s in range(SEEDS)]))
            conn = float(np.mean([r[3] for r in cur]))
            w_conn = float(np.clip(w_conn + 0.5 * (CONN_TARGET - conn), 0.1, 5.0))
    return theta, w_conn


def evaluate(theta, mode, steps, scales, phase):
    out = []
    for (grid, n) in scales:
        pol = make_fiedler_policy(theta, max(1, n - 2)) if mode == "fiedler" else make_theta_policy(theta, max(1, n - 2))
        acc = np.zeros(4)
        for s in range(3):
            m = rollout_mission(grid, grid, random_wall(grid, grid, 0.05, 300 + s), n, 5, 3,
                                steps[grid], 300 + s, pol, backstop=False, decay=0.97)
            acc += [m["delivered_cov"], m["connectivity"], m["safety_viol"], m["relay_frac"]]
        acc /= 3
        out.append(dict(grid=grid, n=n, phase=phase, cov=round(acc[0], 3), conn=round(acc[1], 3),
                        safety=round(acc[2], 3), relay=round(acc[3], 3)))
    return out


def main():
    workers = min(mp.cpu_count(), 24)
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    rng = np.random.default_rng(0)
    log = open(os.path.join(OUT, "relay_experiments.log"), "w")

    def say(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()

    experiments = [("baseline", "standard", STD, False), ("adaptive", "standard", STD, True),
                   ("fiedler", "fiedler", STD, False), ("episodes", "standard", LONG, False)]
    results = {}
    say(f"Generalization study on {workers} cores · TRAIN {TRAIN} · ZERO-SHOT {ZEROSHOT} · FEW-SHOT {FEWSHOT}")
    try:
        for name, mode, steps, adaptive in experiments:
            theta, w_final = es_train(mode, steps, adaptive, TRAIN, GENS, pool, rng)
            rows = evaluate(theta, mode, steps, TRAIN, "train") + evaluate(theta, mode, steps, ZEROSHOT, "zero-shot")
            # few-shot: adapt the trained policy a few gens on the held-out FEWSHOT size
            theta_fs, _ = es_train(mode, steps, False, [FEWSHOT], FEWSHOT_GENS, pool, rng, theta0=theta)
            rows += evaluate(theta_fs, mode, steps, [FEWSHOT], "few-shot")
            results[name] = dict(mode=mode, w_conn_final=round(w_final, 2), rows=rows)
            say(f"[{name}] w_conn_final={w_final:.2f}")
            for p in rows:
                say(f"   {p['phase']:9s} {p['grid']}²/N{p['n']}: cov {p['cov']*100:3.0f}%  conn {p['conn']*100:3.0f}%  "
                    f"safety {p['safety']:.3f}  relay {p['relay']*100:3.0f}%")
            json.dump(results, open(os.path.join(OUT, "experiments.json"), "w"), indent=2)
    finally:
        pool.shutdown()
    say("=== done ===")


if __name__ == "__main__":
    main()
