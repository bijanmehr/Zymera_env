"""Evolution (OpenAI-ES) of the DEGREE-SIGNAL policy.

The agent's role (frontier vs return-to-herd) is decided by a learned policy over
degree features -- normalized degree, neighbors lost from its own peak, detached
flag, gossiped component fraction (global-split awareness), local unexplored
density, coverage progress. ES discovers the role-switch rule instead of a fixed
floor, so relay/frontier specialization can EMERGE. Decentralized (shared policy);
global team fitness = (1-w)*coverage + w*largest_comp_frac.

  PYTHONPATH=. .venv/bin/python -m swarm_explore.evolve_degree --grid 32 --agents 10
  PYTHONPATH=. .venv/bin/python -m swarm_explore.evolve_degree --grid 16 --agents 4
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


def _metrics(theta, H, W, wall, N, cr, sr, steps, seeds):
    cov = conn = lg = 0.0
    for s in seeds:
        r = run_swarm(H, W, wall, N, cr, sr, steps, s, degree_theta=theta)
        cov += r["coverage"]; conn += r["connectivity"]; lg += r["largest_comp_frac"]
    n = len(seeds)
    return cov / n, conn / n, lg / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=14)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--alpha", type=float, default=0.08)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--agents", type=int, default=10)
    ap.add_argument("--comm-r", type=int, default=5)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--w-conn", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm", type=str, default="")
    a = ap.parse_args()
    H = W = a.grid
    sr = 3
    wall = random_wall(H, W, 0.05, 0)
    seeds = list(range(a.eval_seeds))
    rng = np.random.default_rng(a.seed)

    # reference: the fixed-floor degree rule (K=7) on seed 0, for context
    _r = run_swarm(H, W, wall, a.agents, a.comm_r, sr, a.steps, seeds[0],
                   degree_floor=max(1, (a.agents - 1) - 7))
    print(f"ref fixed-floor (K=7) seed0: cov {_r['coverage']*100:.0f}%  conn {_r['connectivity']*100:.0f}%  "
          f"largest {_r['largest_comp_frac']*100:.0f}%", flush=True)
    print(f"ES degree-policy: {a.gens} gens x {2*a.pop} pop x {a.eval_seeds} seeds @ "
          f"{H}x{H}/{a.agents}/cr{a.comm_r}, w_conn={a.w_conn}" + (f", warm={a.warm}" if a.warm else ""), flush=True)

    theta = np.load(a.warm) if (a.warm and os.path.exists(a.warm)) else P.init_degree_theta(a.seed)

    def fit(th):
        c, k, lg = _metrics(th, H, W, wall, a.agents, a.comm_r, sr, a.steps, seeds)
        return (1 - a.w_conn) * c + a.w_conn * lg

    best = (fit(theta), theta.copy())
    hist = []
    for g in range(a.gens):
        eps = rng.normal(0, 1, (a.pop, theta.size))
        eps = np.concatenate([eps, -eps], 0)
        F = np.array([fit(theta + a.sigma * e) for e in eps])
        A = (F - F.mean()) / (F.std() + 1e-8)
        theta = theta + a.alpha * (eps.T @ A) / (len(eps) * a.sigma)
        cov, conn, lg = _metrics(theta, H, W, wall, a.agents, a.comm_r, sr, a.steps, seeds)
        f = (1 - a.w_conn) * cov + a.w_conn * lg
        hist.append(dict(gen=g, fit=float(f), cov=float(cov), conn=float(conn), largest=float(lg)))
        print(f"  gen {g:2d}  fit {f:.3f}  cover {cov*100:5.1f}%  conn {conn*100:4.0f}%  largest {lg*100:4.0f}%", flush=True)
        if f > best[0]:
            best = (f, theta.copy())

    cov, conn, lg = _metrics(best[1], H, W, wall, a.agents, a.comm_r, sr, a.steps, seeds)
    print(f"\nBEST degree-policy: cov {cov*100:.1f}%  conn {conn*100:.0f}%  largest {lg*100:.0f}%", flush=True)

    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "experiments",
                                        "swarm_explore",
                                        "degree-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "best_degree.npy"), best[1])
    with open(os.path.join(out, "degree.json"), "w") as f:
        json.dump({"config": vars(a), "best": [cov, conn, lg], "history": hist}, f, indent=2)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
