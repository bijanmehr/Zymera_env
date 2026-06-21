"""Run the experiment suite (Tier-0 reference frame + headline + tension sweep)
and save an honest report. v1 Mission = deterministic uncertainty compass
(decentralized); the learned compass is the next build step.

  PYTHONPATH=. .venv/bin/python -m swarm_explore.run --grid 16 --agents 4 --steps 100
"""
from __future__ import annotations

import argparse
import datetime
import json
import os

import numpy as np

from . import baselines
from .core import random_wall
from .swarm import run_swarm


def avg_over_seeds(fn, seeds, **kw):
    keys = ("coverage", "connectivity", "evenness", "redundancy", "dol_gini", "t_cov")
    acc = {k: [] for k in keys}
    for s in seeds:
        m = fn(seed=s, **kw)
        for k in keys:
            acc[k].append(m[k])
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}


def fmt(row):
    return (f"{row['coverage'][0]*100:5.1f}% {row['t_cov'][0]:6.1f} {row['connectivity'][0]*100:6.0f}% "
            f"{row['evenness'][0]:7.2f} {row['redundancy'][0]:7.2f} {row['dol_gini'][0]:6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--comm-r", type=int, default=5)
    ap.add_argument("--sense-r", type=int, default=3)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--obstacle-density", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--lambda-conn", type=float, default=1.0)
    args = ap.parse_args()

    H = W = args.grid
    wall = random_wall(H, W, args.obstacle_density, 0)
    seeds = list(range(args.seeds))
    common = dict(H=H, W=W, wall=wall, n_agents=args.agents, comm_r=args.comm_r,
                  sense_r=args.sense_r, steps=args.steps)

    print(f"grid {H}x{W} | {args.agents} agents | comm_r {args.comm_r} | sense_r {args.sense_r} "
          f"| {args.steps} steps | obstacles {args.obstacle_density:.0%} | {args.seeds} seeds")
    print(f"{'condition':26s} {'cover':>6s} {'T_cov':>6s} {'conn':>7s} {'even':>7s} {'redund':>7s} {'DoL':>6s}")
    print("-" * 74)

    results = {}

    # --- Tier 0: reference frame ---
    results["single agent"] = avg_over_seeds(
        lambda seed, **kw: run_swarm(seed=seed, lambda_conn=0.0, **kw),
        seeds, H=H, W=W, wall=wall, n_agents=1, comm_r=args.comm_r,
        sense_r=args.sense_r, steps=args.steps)
    results["random walk"] = avg_over_seeds(baselines.random_walk, seeds, **common)
    results["lawnmower (ceiling*)"] = avg_over_seeds(baselines.lawnmower, seeds, **common)

    # --- Headline: decentralized swarm ---
    results["swarm (decentralized)"] = avg_over_seeds(
        lambda seed, **kw: run_swarm(seed=seed, lambda_conn=args.lambda_conn, **kw),
        seeds, **common)

    for name, row in results.items():
        print(f"{name:26s} {fmt(row)}")
    print("* lawnmower is omniscient/centralized — efficiency ceiling, not a valid swarm.")

    # --- Experiment 3: coverage<->connectivity tension sweep ---
    print(f"\nTension sweep (lambda_conn): coverage / connectivity / T_cov")
    sweep = {}
    for lam in (0.0, 0.5, 1.0, 2.0, 4.0):
        row = avg_over_seeds(
            lambda seed, **kw: run_swarm(seed=seed, lambda_conn=lam, **kw), seeds, **common)
        sweep[lam] = row
        print(f"  lambda={lam:<4.1f}  cover {row['coverage'][0]*100:5.1f}%   "
              f"conn {row['connectivity'][0]*100:5.0f}%   T_cov {row['t_cov'][0]:5.1f}")

    # --- save ---
    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "experiments",
                                        "swarm_explore",
                                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    os.makedirs(out, exist_ok=True)
    summary = {"config": vars(args),
               "results": {k: {m: v for m, v in row.items()} for k, row in results.items()},
               "tension_sweep": {str(l): {m: v for m, v in row.items()} for l, row in sweep.items()}}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved -> {out}/summary.json")

    try:                                       # viz must never break the numbers
        from . import viz
        hl = run_swarm(seed=0, lambda_conn=args.lambda_conn, record=True, **common)
        viz.territory_map(hl, os.path.join(out, "territories.png"), title="swarm: who covered what")
        viz.certainty_heatmap(hl, os.path.join(out, "certainty.png"), title="certainty field")
        viz.pareto(sweep, os.path.join(out, "pareto.png"))
        print(f"saved viz -> {out}/territories.png, certainty.png, pareto.png")
    except Exception as e:
        print(f"[viz skipped: {e}]")
    return results, sweep


if __name__ == "__main__":
    main()
