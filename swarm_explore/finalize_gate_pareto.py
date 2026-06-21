"""Assemble the LEARNED relay-gate Pareto at 32x32 and redraw the Pareto figure.

Evaluates, at a common 5-seed protocol: the 16x16-trained gate zero-shot, plus
every warm-retrained 32x32 gate (the w_conn sweep). Writes the points into
scale/results.json under 'learned_32' and overlays them on pareto_32.png next to
the baseline soft-lambda sweep and the heuristic relay points.

  PYTHONPATH=. .venv/bin/python -m swarm_explore.finalize_gate_pareto
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

from .core import random_wall          # noqa: E402
from .swarm import run_swarm           # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SCALE = os.path.join(ROOT, "experiments", "swarm_explore", "scale")
SEEDS = list(range(5))


def eval_gate(gate, H=32, N=10):
    acc = dict(coverage=0.0, connectivity=0.0, largest_comp_frac=0.0, relay_frac=0.0)
    for s in SEEDS:
        wall = random_wall(H, H, 0.05, s)
        r = run_swarm(H, H, wall, N, 5, 3, 100, s, topo_gossip=True, relay=True,
                      relay_margin=0, gate_theta=gate)
        for k in acc:
            acc[k] += r[k]
    return {k: v / len(SEEDS) for k, v in acc.items()}


def _find(grid, warm):
    hits = []
    for f in sorted(glob.glob(os.path.join(ROOT, "experiments/swarm_explore/gate-*/gate.json"))):
        d = json.load(open(f))
        c = d["config"]
        if c["grid"] == grid and bool(c.get("warm")) == warm:
            hits.append((f, d))
    return hits


def main():
    learned = []

    # 16x16-trained gate, zero-shot at 32x32
    g16 = _find(16, False)
    if g16:
        path = os.path.join(os.path.dirname(g16[-1][0]), "best_gate.npy")
        m = eval_gate(np.load(path))
        m.update(label="gate16 zero-shot", w_conn=None)
        learned.append(m)
        print(f"  gate16 zero-shot @32   cov {m['coverage']*100:5.1f}%  conn {m['connectivity']*100:4.0f}%  "
              f"largest {m['largest_comp_frac']*100:4.0f}%", flush=True)

    # warm-retrained 32x32 gates (w_conn sweep)
    for f, d in _find(32, True):
        path = os.path.join(os.path.dirname(f), "best_gate.npy")
        w = d["config"]["w_conn"]
        m = eval_gate(np.load(path))
        m.update(label=f"learned gate (w={w:g})", w_conn=w)
        learned.append(m)
        print(f"  learned gate w={w:g} @32  cov {m['coverage']*100:5.1f}%  conn {m['connectivity']*100:4.0f}%  "
              f"largest {m['largest_comp_frac']*100:4.0f}%", flush=True)

    # merge into results.json
    res_path = os.path.join(SCALE, "results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    results["learned_32"] = learned
    json.dump(results, open(res_path, "w"), indent=2)

    # redraw Pareto. baseline = one stuck point (any lambda). relay frontier =
    # the robust, tunable curve from relay aggressiveness: cut-vertex -> +leaf-edge
    # -> clump corner. learned gates overlaid as markers.
    base = results.get("baseline_32", [])
    relay = results.get("relay_32", [])
    abl = results.get("ablations_32", {})
    b1 = next((m for m in base if m["lambda"] == 1.0), base[0] if base else None)
    r1 = next((m for m in relay if m["lambda"] == 1.0), relay[0] if relay else None)
    clump = next((m for m in learned if m.get("w_conn") == 0.9), None)  # all-relay extreme

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for axi, (key, ttl) in zip(ax, [("connectivity", "strict full-connectivity"),
                                    ("largest_comp_frac", "largest-component fraction (cohesion)")]):
        # relay frontier (robust, tunable): baseline -> cut-vertex -> leaf-edge -> clump
        frontier = [m for m in [b1, r1, abl.get("relay_leafedge"), clump] if m]
        frontier = sorted(frontier, key=lambda m: -m["coverage"])
        axi.plot([m["coverage"]*100 for m in frontier], [m[key]*100 for m in frontier],
                 "s-", color="#d9480f", label="relay frontier (tunable, robust)")
        if b1:
            axi.scatter([b1["coverage"]*100], [b1[key]*100], marker="o", s=90,
                        color="#9aa0a6", zorder=4, label="baseline (any λ — one stuck point)")
        if learned:
            for m in learned:
                tag = "gate16 zero-shot" if m["w_conn"] is None else f"learned w={m['w_conn']:g}"
                axi.scatter([m["coverage"]*100], [m[key]*100], marker="*", s=170,
                            color="#1f6feb", zorder=5)
                axi.annotate(tag, (m["coverage"]*100, m[key]*100), fontsize=7, color="#1f6feb")
        axi.scatter([], [], marker="*", s=120, color="#1f6feb", label="learned relay-gate")
        axi.set_xlabel("coverage  (%)")
        axi.set_ylabel(f"{ttl}  (%)")
        axi.set_title(f"32×32 / 10 — coverage vs {ttl}")
        axi.grid(alpha=0.3)
        axi.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(SCALE, "pareto_32.png"), dpi=110)
    plt.close(fig)
    print(f"\nupdated -> {res_path} and pareto_32.png", flush=True)


if __name__ == "__main__":
    main()
