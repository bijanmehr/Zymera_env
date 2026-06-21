"""Does the DEGREE SIGNAL help the PPO swarm? Trains IPPO and CTDE with vs.
without the two degree observation channels (normalized degree + gossiped
component fraction) on comm-coverage, and compares — an A/B for whether
degree-awareness changes emergent connected coverage.

Sequential (one process) to stay light on CPU. Writes a single run dir with a
comparison.csv tagged by (variant, degree).

  python examples/runners/run_marl_degree.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc
import _marl_core as core
from marl_independent import IndependentAC
from marl_ctde import CTDE_AC

VARIANTS = [("independent", IndependentAC), ("CTDE", CTDE_AC)]
DEGREE = [("base", False), ("degree", True)]

base = core.new_run_dir()
print(f"run dir: {base}\n")
rows = []
for vname, M in VARIANTS:
    for dname, dflag in DEGREE:
        label = f"{vname}-{dname}"
        env = cc.make_env(degree_channels=dflag)
        print(f"\n==== {label}  (obs_channels={env.obs_channels}) ====", flush=True)
        b = core.run(env, M, label, outdir=base)
        rows.append((vname, dname, b))

with open(base / "comparison.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["variant", "degree", "coverage", "connected", "max_dist",
                "redundancy", "collisions", "best_iter"])
    for vname, dname, b in rows:
        w.writerow([vname, dname, f"{b['cov']:.4f}", f"{b['conn']:.4f}", f"{b['maxd']:.4f}",
                    f"{b['redun']:.4f}", f"{b['coll']:.4f}", b["iter"]])

print("\n========== DEGREE-SIGNAL A/B · best per config ==========")
print(f"{'config':22s} {'coverage':>9s} {'connected':>10s} {'max-dist':>9s} {'redun':>7s}")
for vname, dname, b in rows:
    print(f"{vname+'-'+dname:22s} {b['cov']*100:8.1f}% {b['conn']*100:9.0f}% "
          f"{b['maxd']:9.1f} {b['redun']:6.2f}×")
print(f"\nartifacts: {base}")
