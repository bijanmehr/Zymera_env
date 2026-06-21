"""Push CTDE coverage past 90% WITHOUT relaxing the 70-step cap or abandoning
connectivity — by reward-shaping toward the coordinated side-by-side sweep.

The lever is the anti-overlap term (w_overlap): it penalizes an agent for
covering cells another agent covers the SAME step, breaking the clump↔chain
symmetry toward a SPREAD sweep (division of labour → lower redundancy → more
unique coverage). This sweeps a dose-response on w_overlap (0 → 0.5 → 1.0) plus
a coverage-forward variant, all on CTDE, 70 steps, clustered start, and reports
the coverage/connectivity/redundancy frontier.

    python examples/coverage_push.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc
import _marl_core as core
from marl_ctde import CTDE_AC

core.ITERS = 180   # the coordinated sweep is a harder policy to form than clump-cover

CONFIGS = [
    ("baseline",   dict(w_cov=1.0, w_conn=2.0, w_overlap=0.0, w_coll=4.0)),
    ("ov0.5",      dict(w_cov=1.0, w_conn=2.0, w_overlap=0.5, w_coll=4.0)),
    ("ov1.0",      dict(w_cov=1.0, w_conn=2.0, w_overlap=1.0, w_coll=4.0)),
    ("covfwd-ov1", dict(w_cov=2.0, w_conn=1.0, w_overlap=1.0, w_coll=4.0)),
]

base = core.new_run_dir()
print(f"coverage-push sweep · CTDE · 70 steps · clustered · {core.ITERS} iters → {base}\n")
results = {}
for name, w in CONFIGS:
    results[name] = core.run(cc.make_env(**w), CTDE_AC, name, outdir=base)

print("\n================= COVERAGE-PUSH · best per config =================")
print(f"{'config':13s} {'w_cov':>5} {'w_conn':>6} {'w_ovlp':>6} | "
      f"{'coverage':>9} {'connected':>10} {'max-dist':>9} {'redundancy':>11}")
for name, w in CONFIGS:
    b = results[name]
    print(f"{name:13s} {w['w_cov']:5.1f} {w['w_conn']:6.1f} {w['w_overlap']:6.1f} | "
          f"{b['cov']*100:8.1f}% {b['conn']*100:9.0f}% {b['maxd']:9.1f} {b['redun']:10.2f}×")
best = max(results.items(), key=lambda kv: kv[1]["cov"])
print(f"\nBEST coverage: {best[0]} — {best[1]['cov']*100:.1f}%  "
      f"(connected {best[1]['conn']*100:.0f}%, redundancy {best[1]['redun']:.2f}×, "
      f"max-dist {best[1]['maxd']:.1f})")
print(f"artifacts: {base}")
