"""ES training for Stack 2 (role picker + switcher) with the MISSION-SAFETY fitness.

  fitness = coverage  -  w_safety * (degree-budget violations)  +  w_coh * cohesion
  (an agent that has lost > K neighbors, degree < N-1-K, is a safety break).

Gradient-free ES over a shared small policy conditioned on per-agent graph features ->
structure-aware roles emerge from the feature-conditioning + whole-team fitness, with
NO credit-assignment problem. Trained over MANY maps for robustness.

  PYTHONPATH=. .venv/bin/python -m swarm_explore.evolve_switcher --gens 16 --w-safety 1.0 --K 2
"""
from __future__ import annotations

import argparse
import datetime
import json
import os

import numpy as np

from .core import random_wall
from .arch3 import init_theta, run_arch, SW_NPARAMS


def fitness_metrics(theta, H, n, cr, sr, steps, K, hi, lo, seeds):
    cov = conn = coh = sv = rl = 0.0
    for s in seeds:
        wall = random_wall(H, H, 0.05, s)
        m = run_arch(H, H, wall, n, cr, sr, steps, s, theta, K=K, hi=hi, lo=lo)
        cov += m["coverage"]; conn += m["connectivity"]; coh += m["largest_comp_frac"]
        sv += m["safety_viol"]; rl += m["relay_frac"]
    k = len(seeds)
    return dict(cov=cov / k, conn=conn / k, coh=coh / k, sv=sv / k, rl=rl / k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=16)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--alpha", type=float, default=0.08)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--comm-r", type=int, default=5)
    ap.add_argument("--steps", type=int, default=70)
    ap.add_argument("--eval-seeds", type=int, default=6)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--w-safety", type=float, default=1.0)
    ap.add_argument("--w-coh", type=float, default=0.2)
    ap.add_argument("--hi", type=float, default=0.6)
    ap.add_argument("--lo", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()
    H, n, sr = a.grid, a.agents, 3
    seeds = list(range(a.eval_seeds))
    rng = np.random.default_rng(a.seed)
    theta = init_theta(a.seed)

    def fit(th):
        m = fitness_metrics(th, H, n, a.comm_r, sr, a.steps, a.K, a.hi, a.lo, seeds)
        return m["cov"] - a.w_safety * m["sv"] + a.w_coh * m["coh"], m

    print(f"ES switcher [{a.tag}] {a.gens}gen x {2*a.pop}pop x {a.eval_seeds}seed @ {H}x{H}/{n}/{a.steps}st  "
          f"K={a.K} w_safety={a.w_safety} w_coh={a.w_coh}", flush=True)
    bestf, bestm = fit(theta); best = (bestf, theta.copy(), bestm)
    hist = []
    for g in range(a.gens):
        eps = rng.normal(0, 1, (a.pop, theta.size)); eps = np.concatenate([eps, -eps], 0)
        F = np.array([fit(theta + a.sigma * e)[0] for e in eps])
        A_ = (F - F.mean()) / (F.std() + 1e-8)
        theta = theta + a.alpha * (eps.T @ A_) / (len(eps) * a.sigma)
        f, m = fit(theta)
        hist.append(dict(gen=g, fit=float(f), **{k: float(v) for k, v in m.items()}))
        print(f"  gen {g:2d}  fit {f:.3f}  cov {m['cov']*100:5.1f}%  conn {m['conn']*100:4.0f}%  "
              f"coh {m['coh']*100:4.0f}%  safety-viol {m['sv']*100:4.1f}%  relay {m['rl']*100:3.0f}%", flush=True)
        if f > best[0]:
            best = (f, theta.copy(), m)

    m = best[2]
    print(f"\nBEST: cov {m['cov']*100:.1f}%  conn {m['conn']*100:.0f}%  coh {m['coh']*100:.0f}%  "
          f"safety-viol {m['sv']*100:.1f}%  relay {m['rl']*100:.0f}%", flush=True)
    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "experiments", "swarm_explore",
                                        f"switcher-{a.tag}-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "best_theta.npy"), best[1])
    json.dump({"config": vars(a), "best": {k: float(v) for k, v in m.items()}, "history": hist},
              open(os.path.join(out, "switcher.json"), "w"), indent=2)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
