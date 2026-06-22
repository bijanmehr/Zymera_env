#!/usr/bin/env python
"""Train ONE MARL variant at ONE scale; write a result JSON.

Run as a SUBPROCESS by the method-tournament job: the MARL core reads scale from
module globals at import time, so each (variant, scale) needs a fresh process.
Generalizes examples/runners/run_marl_32.py to all variants + JSON output.

  python -m experiments.marl_tournament <variant> <grid> <nag> <steps> <envs> <iters> <seed> <outdir>
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "lib"))     # so _marl_core / comm_coverage / models import

variant = sys.argv[1]
grid, nag, steps, envs, iters, seed = (int(sys.argv[i]) for i in range(2, 8))
outdir = sys.argv[8]

# --- set scale on the env module BEFORE importing the core (it reads cc.* at import) ---
import comm_coverage as cc                            # noqa: E402
cc.GRID = grid; cc.N_AGENTS = nag; cc.MAX_STEPS = steps; cc.COMM_R = 5
import _marl_core as core                             # noqa: E402
core.ENVS = envs; core.ITERS = iters; core.T = steps  # PPO batch + length + horizon
import jax                                            # noqa: E402

from marl_independent import IndependentAC            # noqa: E402
from marl_ctde import CTDE_AC                          # noqa: E402
from marl_joint import JointAC                         # noqa: E402
MODELS = {"independent": IndependentAC, "CTDE": CTDE_AC, "joint": JointAC}
try:
    from marl_frontier_attn import FrontierAttnAC      # noqa: E402
    MODELS["frontier-attn"] = FrontierAttnAC
except Exception as e:                                  # frontier model may want a different obs — isolate it
    print(f"(frontier-attn unavailable: {e})", flush=True)

# transformer policies + size sweep (DIM, DEPTH): attn=full self-attention (centralized),
# gattn=masked graph-attention over comm-graph neighbours (decentralized, connectivity-aware)
ATTN_SIZES = {"attn-s": (32, 1), "attn": (64, 2), "attn-m": (96, 2), "attn-l": (128, 3)}
GATTN_SIZES = {"gattn-s": (32, 1), "gattn": (64, 2), "gattn-m": (96, 2), "gattn-l": (128, 3)}
SIZE_MAP = {**ATTN_SIZES, **GATTN_SIZES}
try:
    import marl_attn                                    # noqa: E402
    from marl_attn import AgentAttnAC, GraphAttnAC      # noqa: E402
    for _name in ATTN_SIZES:
        MODELS[_name] = AgentAttnAC
    for _name in GATTN_SIZES:
        MODELS[_name] = GraphAttnAC
except Exception as e:
    print(f"(attn unavailable: {e})", flush=True)

# the proven 32x32 reward recipe (run_marl_32.py), parameterized by scale:
# spread + light capped connectivity + degree signal in the obs (policy self-regulates the comm graph)
ENV_KW = dict(
    grid_h=grid, grid_w=grid, n_agents=nag, comm_r=5,
    vis_r=3, sense_r=3, n_obstacles=max(8, grid * grid * 3 // 100), spawn_radius=2,
    w_cov=1.0, w_conn=0.5, conn_cap=nag - 1, w_coll=4.0, w_overlap=1.0,
    w_shape1=1.0, w_shape3=1.0, shape_gamma=0.99, w_degree=0.2, degree_floor=1,
    degree_channels=True,
)


def main():
    if variant not in MODELS:
        print(f"unknown/unavailable variant {variant}", flush=True); sys.exit(2)
    if variant in SIZE_MAP:                             # set transformer size for this variant
        marl_attn.DIM, marl_attn.DEPTH = SIZE_MAP[variant]
    env = cc.make_env(**ENV_KW)
    label = f"{variant}-{grid}x{grid}-n{nag}-s{seed}"
    os.makedirs(outdir, exist_ok=True)
    b = core.run(env, MODELS[variant], label, seed=seed, outdir=outdir,
                 target_cov=0.97, patience=60, min_iters=min(50, iters - 1))
    rec = dict(variant=variant, grid=grid, n=nag, seed=seed,
               cov=round(b["cov"], 4), conn=round(b["conn"], 4), maxd=round(b["maxd"], 3),
               redun=round(b["redun"], 3), coll=round(b["coll"], 3), iter=b["iter"])
    json.dump(rec, open(os.path.join(outdir, f"result_{label}.json"), "w"), indent=2)
    print("RESULT " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
