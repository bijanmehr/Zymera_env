"""Run all three MARL PPO variants on the swarm comm-coverage task and print a
comparison of their best results. (independent / CTDE / joint — differ only in
actor/critic centralization; shared core + reward + regularization.)

All three variants of one invocation write into a single timestamped run dir,
experiments/marl/<timestamp>/{independent,ctde,joint}/, plus a top-level
comparison.csv. Each variant dir holds metrics.npz + checkpoint.eqx + config.json."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc
import _marl_core as core
from marl_independent import IndependentAC
from marl_ctde import CTDE_AC
from marl_joint import JointAC

VARIANTS = [("independent", IndependentAC), ("CTDE", CTDE_AC), ("joint", JointAC)]

base = core.new_run_dir()
print(f"run dir: {base}\n")
results = {}
for name, M in VARIANTS:
    results[name] = core.run(cc.make_env(), M, name, outdir=base)

# machine-readable cross-variant summary
with open(base / "comparison.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["variant", "coverage", "connected", "max_dist",
                "redundancy", "collisions", "best_iter"])
    for name, b in results.items():
        w.writerow([name, f"{b['cov']:.4f}", f"{b['conn']:.4f}", f"{b['maxd']:.4f}",
                    f"{b['redun']:.4f}", f"{b['coll']:.4f}", b["iter"]])

print("\n================ COMPARISON · best per variant ================")
print(f"{'variant':12s} {'coverage':>9s} {'connected':>10s} {'max-dist':>9s} "
      f"{'redundancy':>11s} {'collisions':>11s}")
for name, b in results.items():
    print(f"{name:12s} {b['cov']*100:8.1f}% {b['conn']*100:9.0f}% {b['maxd']:9.1f} "
          f"{b['redun']:10.2f}× {b['coll']:11.2f}")
print(f"(a stretched chain shows as connected≈100% AND max-dist > comm range {cc.COMM_R})")

best = max(results.items(), key=lambda kv: kv[1]["cov"])
print(f"\nBEST coverage: {best[0]} — {best[1]['cov']*100:.1f}%  "
      f"(connected {best[1]['conn']*100:.0f}%, max-dist {best[1]['maxd']:.1f}, "
      f"redundancy {best[1]['redun']:.2f}×, collisions {best[1]['coll']:.2f})")
print(f"\nartifacts: {base}  (per-variant metrics.npz/checkpoint.eqx/config.json · comparison.csv)")
