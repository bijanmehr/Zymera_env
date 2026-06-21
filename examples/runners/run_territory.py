"""De-confliction test #1 (minimal): give each agent a "this is your share" channel
(the uncovered cells it's the nearest in-range agent to) and let its existing sweeper
cover its own territory. 2 agents / 8x16 / 70 steps / no connectivity.

Baseline (no territory) was 80.5%, redundancy 1.09x. Single agent on its quarter = 93%.
Hypothesis: territory -> coverage up toward ~90%, redundancy down toward 1.0.

    python -u examples/run_territory.py
"""
import statistics as stt
import sys
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc

cc.N_AGENTS = 2
cc.GRID = 12
import _marl_core as core          # noqa: E402
import _frontier_core as fc        # noqa: E402
from marl_frontier_comm import FrontierCommAttnAC as M  # noqa: E402

STEPS, N, GH, GW = 70, 2, 8, 16
CELLS = GH * GW
cc.MAX_STEPS = STEPS
core.T = STEPS
fc.T = STEPS
core.ITERS = 200
ENV = dict(grid_h=GH, grid_w=GW, n_agents=N, comm_r=5, spawn_radius=2)
SHAPE = dict(w_shape1=1.0, w_shape3=2.0)
SEEDS = [1, 2, 3]


def evalp(env, build, vdir):
    m = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"), build(env, jax.random.PRNGKey(0)))
    tr = jax.vmap(lambda k: fc._rollout_masked(env, m, STEPS, k))(
        jax.random.split(jax.random.PRNGKey(7), 32))
    st = tr["world"]
    te = st.team_explored[:, -1].reshape(32, -1)
    cov = float((te.sum(-1) / CELLS).mean())
    indiv = st.explored_by[:, -1].reshape(32, N, -1).sum((-1, -2))
    redun = float((indiv / jnp.maximum(te.sum(-1), 1)).mean())
    fresh = cov * CELLS / (N * (STEPS + 1))
    d = jnp.max(jnp.abs(st.pos[:, -1, :, None, :] - st.pos[:, -1, None, :, :]), -1)
    maxd = float(d.max((-1, -2)).mean())
    return cov, redun, fresh, maxd


base = core.new_run_dir()
print(f"TERRITORY de-confliction · 2 agents / {GH}x{GW} ({CELLS} cells) / {STEPS} steps · {base}\n", flush=True)
VARIANTS = [
    ("territory", dict(territory_channel=True), dict()),
]  # baseline already known: 80.5% +/-0.4, redun 1.09x (run_pair)
for name, ekw, mkw in VARIANTS:
    build = partial(M, **mkw)
    res = []
    for sd in SEEDS:
        env = cc.make_env(w_cov=1.0, w_conn=0.0, w_degree=0.0, **SHAPE, **ekw, **ENV)
        b = fc.run(env, build, f"{name}-s{sd}", outdir=base, seed=sd,
                   target_cov=None, patience=999, min_iters=999, mask_collisions=False)
        res.append(evalp(env, build, Path(b["dir"])))
    covs = [r[0] for r in res]
    print(f"  ===> {name:10s} cov {stt.mean(covs)*100:5.1f}% ±{stt.pstdev(covs)*100:3.1f}  "
          f"redun {stt.mean([r[1] for r in res]):.2f}x  fresh {stt.mean([r[2] for r in res])*100:3.0f}%  "
          f"maxd {stt.mean([r[3] for r in res]):.1f}  seeds={[round(c*100, 1) for c in covs]}", flush=True)
print(f"\n  refs: 1 agent/quarter=93%, 2-agent baseline=80.5%/1.09x. territory should beat both if de-confliction is the lever.")
print(f"artifacts: {base}", flush=True)
