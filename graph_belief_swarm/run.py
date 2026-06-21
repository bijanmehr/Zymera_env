"""Entry point: run the headline emergence ablation and save visualizations.

  full (merge + anti-crowding)  -- the designed system
  no anti-crowding (herd)        -- predicted to clump / re-cover (herd effect)
  no merge (independent)         -- agents don't share belief

Run (from the zymera_env project root, with the project venv):

  PYTHONPATH=. .venv/bin/python -m graph_belief_swarm.run --grid 20 --agents 4 --steps 200
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .swarm import run_episode


def _random_wall(H, W, density, seed):
    rng = np.random.default_rng(seed + 777)
    wall = rng.random((H, W)) < density
    wall[0, 0] = False                    # keep a guaranteed free anchor
    return wall


def main():
    ap = argparse.ArgumentParser(description="Graph-belief swarm prototype")
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--agents", type=int, default=5)
    ap.add_argument("--obstacle-density", type=float, default=0.05)
    ap.add_argument("--comm-r", type=int, default=6)
    ap.add_argument("--sense-r", type=int, default=3)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--tether", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    H = W = args.grid
    wall = _random_wall(H, W, args.obstacle_density, args.seed)
    A = args.agents

    print(f"grid {H}x{W} | {A} agents | comm_r {args.comm_r} | "
          f"sense_r {args.sense_r} | {args.steps} steps | seed {args.seed}")

    # The bar to beat: one agent, same time budget.
    single = run_episode(H, W, wall, 1, args.comm_r, args.sense_r,
                         args.steps, args.seed, strategy="sweep")
    single_cov = single["coverage"]

    conditions = [
        ("1 agent (sweep)",          single, None),
        ("connected swarm (sweep)",  None, dict(strategy="sweep")),
        ("swarm: greedy compass",    None, dict(merge=True, anti_crowding=True, keep_connected=True,
                                                shield="soft", tether_radius=args.tether)),
    ]

    print(f"{'condition':32s} {'mapped':>7s} {'visited':>8s} {'x single':>9s} "
          f"{'connect':>8s} {'redund':>7s}")
    print("-" * 80)

    results, labels = [], []
    for name, pre, kw in conditions:
        r = pre if pre is not None else run_episode(
            H, W, wall, A, args.comm_r, args.sense_r, args.steps, args.seed,
            record=(kw is not None), **kw)             # record frames for both swarms
        ratio = r["coverage"] / single_cov if single_cov > 0 else float("nan")
        results.append(r); labels.append(name)
        print(f"{name:32s} {r['observed_coverage']*100:6.1f}% {r['coverage']*100:7.1f}% "
              f"{ratio:8.2f}x {r['connectivity']*100:7.0f}% {r['redundancy']:6.2f}x")

    # --- save / checkpoint the run for later data + viz regeneration --------
    from . import persist
    config = dict(grid=H, agents=A, obstacle_density=args.obstacle_density,
                  comm_r=args.comm_r, sense_r=args.sense_r, steps=args.steps,
                  seed=args.seed, tether=args.tether, single_cov=float(single_cov),
                  conditions=labels)
    base = args.out or os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "experiments", "graph_belief_swarm"))
    run_dir = persist.save_run(base, config, list(zip(labels, results)))
    print(f"\nsaved run -> {run_dir}")
    try:
        from . import viz
        viz.render_curves(results, labels, os.path.join(run_dir, "coverage_curves.png"))
        viz.render_territories(results[1], os.path.join(run_dir, "territories_connected.png"),
                               title=labels[1])
        print(f"saved viz  -> {run_dir}/coverage_curves.png, territories_connected.png")
    except Exception as e:           # viz must never break the metrics run
        print(f"[viz skipped: {e}]")


if __name__ == "__main__":
    main()
