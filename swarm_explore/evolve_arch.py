"""ES training for the backbone architecture heads:
  v1  --arch gnn  : the GNN role head (learns when a leaf should flip to relay)
  v2  --arch attn : the attention goal head (learns which frontier a leaf picks)

Both run on the fast host-side swarm engine, so 16x16/4 ES converges in minutes.
Fitness favors coverage (the backbone already over-connects): (1-w)*cov + w*cohesion.

  PYTHONPATH=. .venv/bin/python -m swarm_explore.evolve_arch --arch gnn
  PYTHONPATH=. .venv/bin/python -m swarm_explore.evolve_arch --arch attn
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


def _metrics(theta, arch, H, N, cr, sr, steps, seeds):
    cov = conn = lg = 0.0
    for s in seeds:
        wall = random_wall(H, H, 0.05, s)
        kw = dict(backbone=True)
        kw["gnn_theta" if arch == "gnn" else "attn_theta"] = theta
        r = run_swarm(H, H, wall, N, cr, sr, steps, s, **kw)
        cov += r["coverage"]; conn += r["connectivity"]; lg += r["largest_comp_frac"]
    n = len(seeds)
    return cov / n, conn / n, lg / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["gnn", "attn"], required=True)
    ap.add_argument("--gens", type=int, default=14)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--alpha", type=float, default=0.08)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--comm-r", type=int, default=5)
    ap.add_argument("--steps", type=int, default=70)
    ap.add_argument("--eval-seeds", type=int, default=3)
    ap.add_argument("--w-conn", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    H, sr = a.grid, 3
    seeds = list(range(a.eval_seeds))
    rng = np.random.default_rng(a.seed)
    theta = (P.init_gnn_theta if a.arch == "gnn" else P.init_attn_theta)(a.seed)

    print(f"ES {a.arch} (backbone v{'1' if a.arch=='gnn' else '2'}): {a.gens}gen x {2*a.pop}pop x "
          f"{a.eval_seeds}seed @ {H}x{H}/{a.agents}/cr{a.comm_r}/{a.steps}st  w_conn={a.w_conn}", flush=True)

    def fit(th):
        c, k, lg = _metrics(th, a.arch, H, a.agents, a.comm_r, sr, a.steps, seeds)
        return (1 - a.w_conn) * c + a.w_conn * lg

    best = (fit(theta), theta.copy())
    hist = []
    for g in range(a.gens):
        eps = rng.normal(0, 1, (a.pop, theta.size))
        eps = np.concatenate([eps, -eps], 0)
        F = np.array([fit(theta + a.sigma * e) for e in eps])
        A = (F - F.mean()) / (F.std() + 1e-8)
        theta = theta + a.alpha * (eps.T @ A) / (len(eps) * a.sigma)
        cov, conn, lg = _metrics(theta, a.arch, H, a.agents, a.comm_r, sr, a.steps, seeds)
        f = (1 - a.w_conn) * cov + a.w_conn * lg
        hist.append(dict(gen=g, fit=float(f), cov=float(cov), conn=float(conn), largest=float(lg)))
        print(f"  gen {g:2d}  fit {f:.3f}  cov {cov*100:5.1f}%  conn {conn*100:4.0f}%  cohesion {lg*100:4.0f}%", flush=True)
        if f > best[0]:
            best = (f, theta.copy())

    cov, conn, lg = _metrics(best[1], a.arch, H, a.agents, a.comm_r, sr, a.steps, seeds)
    print(f"\nBEST {a.arch}: cov {cov*100:.1f}%  conn {conn*100:.0f}%  cohesion {lg*100:.0f}%", flush=True)
    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "experiments", "swarm_explore",
                                        f"arch-{a.arch}-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, f"best_{a.arch}.npy"), best[1])
    json.dump({"config": vars(a), "best": [cov, conn, lg], "history": hist},
              open(os.path.join(out, f"{a.arch}.json"), "w"), indent=2)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
