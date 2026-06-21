"""Scale-cooperation experiments (32x32). Answers: does topology-gossip + active
relay-positioning move the coverage<->connectivity Pareto OUTWARD vs. just cranking
the soft lambda knob? Plus ablations (gossip on/off, relay on/off) and a 16x16
sanity check that the mechanism doesn't wreck the regime where it already worked.

  PYTHONPATH=. .venv/bin/python -m swarm_explore.scale_experiments

Writes results JSON + a Pareto PNG under experiments/swarm_explore/scale/.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

from .core import random_wall          # noqa: E402
from .swarm import run_swarm           # noqa: E402

SEEDS = list(range(5))
STEPS = 100
SENSE_R = 3
COMM_R = 5


def _avg(H, N, **kw):
    acc = dict(coverage=0.0, connectivity=0.0, largest_comp_frac=0.0,
               n_comp=0.0, relay_frac=0.0, evenness=0.0, dol_gini=0.0)
    for s in SEEDS:
        wall = random_wall(H, H, 0.05, s)
        r = run_swarm(H, H, wall, N, COMM_R, SENSE_R, STEPS, s, **kw)
        for k in acc:
            acc[k] += r[k]
    return {k: v / len(SEEDS) for k, v in acc.items()}


def _row(label, m):
    return (f"  {label:22s} cov {m['coverage']*100:5.1f}%  conn {m['connectivity']*100:4.0f}%  "
            f"largest {m['largest_comp_frac']*100:4.0f}%  ncomp {m['n_comp']:.2f}  "
            f"relay {m['relay_frac']*100:4.0f}%")


def main():
    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "experiments",
                                        "swarm_explore", "scale"))
    os.makedirs(out, exist_ok=True)
    results = {}

    # ---- Phase 0: baseline soft-lambda Pareto at 32x32 ----
    print("=== Phase 0: baseline lambda-sweep @ 32x32/10 ===", flush=True)
    base = []
    for lam in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
        m = _avg(32, 10, lambda_conn=lam)
        m["lambda"] = lam
        base.append(m)
        print(_row(f"baseline lam={lam}", m), flush=True)
    results["baseline_32"] = base

    # ---- Phase 1: gossip + relay heuristic @ 32x32 (does it beat the Pareto?) ----
    print("\n=== Phase 1: gossip+relay heuristic @ 32x32/10 ===", flush=True)
    relay = []
    for lam in [0.0, 1.0, 2.0]:
        m = _avg(32, 10, lambda_conn=lam, topo_gossip=True, relay=True, relay_margin=0)
        m["lambda"] = lam
        relay.append(m)
        print(_row(f"gossip+relay lam={lam}", m), flush=True)
    results["relay_32"] = relay

    # ---- Phase 1b: ablations @ 32x32 ----
    print("\n=== Phase 1b: ablations @ 32x32/10 (lam=1) ===", flush=True)
    abl = {}
    abl["relay_with_gossip"] = _avg(32, 10, lambda_conn=1.0, topo_gossip=True, relay=True, relay_margin=0)
    print(_row("relay + gossip (exact)", abl["relay_with_gossip"]), flush=True)
    abl["relay_no_gossip"] = _avg(32, 10, lambda_conn=1.0, topo_gossip=False, relay=True, relay_margin=0)
    print(_row("relay, NO gossip (2-hop)", abl["relay_no_gossip"]), flush=True)
    abl["gossip_no_relay"] = _avg(32, 10, lambda_conn=1.0, topo_gossip=True, relay=False)
    print(_row("gossip, NO relay", abl["gossip_no_relay"]), flush=True)
    abl["relay_leafedge"] = _avg(32, 10, lambda_conn=1.0, topo_gossip=True, relay=True, relay_margin=1)
    print(_row("relay + leaf-edge", abl["relay_leafedge"]), flush=True)
    results["ablations_32"] = abl

    # ---- Phase 3: 16x16 sanity (mechanism must not wreck the good regime) ----
    print("\n=== Phase 3: 16x16/4 sanity ===", flush=True)
    s16 = {}
    s16["baseline"] = _avg(16, 4, lambda_conn=1.0)
    print(_row("baseline", s16["baseline"]), flush=True)
    s16["gossip_relay"] = _avg(16, 4, lambda_conn=1.0, topo_gossip=True, relay=True, relay_margin=0)
    print(_row("gossip+relay", s16["gossip_relay"]), flush=True)
    results["sanity_16"] = s16

    # ---- Pareto figure (32x32) ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for axi, ( key, ttl) in zip(ax, [("connectivity", "strict full-connectivity"),
                                     ("largest_comp_frac", "largest-component fraction (cohesion)")]):
        bx = [m["coverage"] * 100 for m in base]
        by = [m[key] * 100 for m in base]
        axi.plot(bx, by, "o-", color="#9aa0a6", label="baseline (soft lambda sweep)")
        for m in base:
            axi.annotate(f"λ={m['lambda']:g}", (m["coverage"]*100, m[key]*100),
                         fontsize=7, color="#5f6368")
        rx = [m["coverage"] * 100 for m in relay]
        ry = [m[key] * 100 for m in relay]
        axi.plot(rx, ry, "s-", color="#d9480f", label="gossip + relay")
        for m in relay:
            axi.annotate(f"λ={m['lambda']:g}", (m["coverage"]*100, m[key]*100),
                         fontsize=7, color="#d9480f")
        axi.set_xlabel("coverage  (%)")
        axi.set_ylabel(f"{ttl}  (%)")
        axi.set_title(f"32x32 / 10 agents — coverage vs {ttl}")
        axi.grid(alpha=0.3)
        axi.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pareto_32.png"), dpi=110)
    plt.close(fig)

    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
