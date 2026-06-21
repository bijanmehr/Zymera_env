"""Evolution (OpenAI-ES) training of the learned role policy.

Decentralized execution (all agents share one policy), global team fitness =
(1-w)·coverage + w·connectivity. Gradient-free, pure numpy, runs on the
host-side sim. Prints baseline + per-generation progress; saves the best policy.

  PYTHONPATH=. .venv/bin/python -m swarm_explore.evolve --gens 12 --pop 8
"""
from __future__ import annotations

import argparse
import datetime
import json
import os

import numpy as np

from . import policy as P
from .core import random_wall
from .swarm import run_swarm


def _metrics(theta, H, W, wall, N, cr, sr, steps, seeds, lambda_conn=None):
    cov = conn = 0.0
    for s in seeds:
        kw = dict(theta=theta) if theta is not None else dict(lambda_conn=lambda_conn)
        r = run_swarm(H, W, wall, N, cr, sr, steps, s, **kw)
        cov += r["coverage"]; conn += r["connectivity"]
    return cov / len(seeds), conn / len(seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--pop", type=int, default=8)          # antithetic -> 2*pop evals/gen
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--comm-r", type=int, default=5)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--eval-seeds", type=int, default=2)
    ap.add_argument("--w-conn", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    H = W = a.grid
    sr = 3
    wall = random_wall(H, W, 0.05, 0)
    seeds = list(range(a.eval_seeds))
    rng = np.random.default_rng(a.seed)

    base_cov, base_conn = _metrics(None, H, W, wall, a.agents, a.comm_r, sr, a.steps, seeds, lambda_conn=1.0)
    print(f"baseline (uniform lambda=1): cover {base_cov*100:.1f}%  conn {base_conn*100:.0f}%", flush=True)
    print(f"ES: {a.gens} gens x {2*a.pop} pop x {a.eval_seeds} seeds @ {H}x{H}/{a.agents}/cr{a.comm_r}, "
          f"w_conn={a.w_conn}", flush=True)

    theta = P.init_theta(a.seed)

    def fit(th):
        c, k = _metrics(th, H, W, wall, a.agents, a.comm_r, sr, a.steps, seeds)
        return (1 - a.w_conn) * c + a.w_conn * k

    best = (-1.0, theta.copy())
    hist = []
    for g in range(a.gens):
        eps = rng.normal(0, 1, (a.pop, theta.size))
        eps = np.concatenate([eps, -eps], 0)               # antithetic pairs
        F = np.array([fit(theta + a.sigma * e) for e in eps])
        A = (F - F.mean()) / (F.std() + 1e-8)              # rank/standardize
        theta = theta + a.alpha * (eps.T @ A) / (len(eps) * a.sigma)
        cov, conn = _metrics(theta, H, W, wall, a.agents, a.comm_r, sr, a.steps, seeds)
        f = (1 - a.w_conn) * cov + a.w_conn * conn
        hist.append(dict(gen=g, fit=float(f), cov=float(cov), conn=float(conn)))
        print(f"  gen {g:2d}  fit {f:.3f}  cover {cov*100:5.1f}%  conn {conn*100:4.0f}%", flush=True)
        if f > best[0]:
            best = (f, theta.copy())

    cov, conn = _metrics(best[1], H, W, wall, a.agents, a.comm_r, sr, a.steps, seeds)
    print(f"\nBEST learned policy: cover {cov*100:.1f}%  conn {conn*100:.0f}%   "
          f"(baseline {base_cov*100:.0f}%/{base_conn*100:.0f}%)", flush=True)

    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "experiments",
                                        "swarm_explore",
                                        "evolve-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "best_theta.npy"), best[1])
    with open(os.path.join(out, "evolve.json"), "w") as f:
        json.dump({"config": vars(a), "baseline": [base_cov, base_conn],
                   "best": [cov, conn], "history": hist}, f, indent=2)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
