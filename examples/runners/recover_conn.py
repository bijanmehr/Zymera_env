"""Two things at once, all with return normalization:

  #2 ABLATION — CTDE + return-norm (same ov0.5 reward as the 91.4% CTDE run).
     Isolates how much of FrontierAttn's 91→98% jump is the return-norm gradient
     fix vs the frontier attention itself.

  #1 RECOVER CONNECTIVITY — FrontierAttn with w_conn raised (5, 8), w_overlap=0.5
     kept. Maps the coverage↔connectivity frontier now that coverage is cheap for
     this net; goal is to buy back the connected swarm while staying >90%.

    python examples/recover_conn.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc
import _marl_core as core
import _frontier_core as fc
from marl_frontier_attn import FrontierAttnAC
from marl_ctde import CTDE_AC

core.ITERS = 180

CONFIGS = [
    ("ctde-returnnorm", CTDE_AC,        dict(w_overlap=0.5, w_conn=2.0)),   # #2 ablation
    ("frontier-conn5",  FrontierAttnAC, dict(w_overlap=0.5, w_conn=5.0)),   # #1
    ("frontier-conn8",  FrontierAttnAC, dict(w_overlap=0.5, w_conn=8.0)),   # #1
]

base = core.new_run_dir()
print(f"recover-conn + ablation · 180 iters · return-norm ON → {base}\n")
results = {}
for name, Model, rw in CONFIGS:
    results[name] = fc.run(cc.make_env(**rw), Model, name, outdir=base)

print("\n============= recover-conn + ablation · best per config =============")
print(f"{'config':18s} {'w_conn':>6} | {'coverage':>9} {'connected':>10} "
      f"{'max-dist':>9} {'redundancy':>11}")
for name, _, rw in CONFIGS:
    b = results[name]
    print(f"{name:18s} {rw['w_conn']:6.1f} | {b['cov']*100:8.1f}% {b['conn']*100:9.0f}% "
          f"{b['maxd']:9.1f} {b['redun']:10.2f}×")
print("\nreference: frontier-attn w_conn=2 → 97.7% cov / 32% conn ;  "
      "CTDE ov0.5 (NO return-norm) → 91.4% cov / 64% conn")
print(f"artifacts: {base}")
