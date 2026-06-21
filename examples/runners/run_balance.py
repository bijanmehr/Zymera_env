"""Exploration-vs-connection tradeoff CURVE. Connectivity (stay together) and coverage
(spread out) are opposed, so instead of guessing a weight, trace the whole exchange:
sweep the connectivity strength (degree-floor weight) from OFF (free spread, max coverage,
low connectivity) to FULL (our 68.9% / 100%-connected point), and read coverage at each
connectivity level. The knee is the balance point to pick.

3 agents / 10x10 / comm_r=5 / 55 steps. Pure cover + (weighted) connectivity, no masking.

    python -u examples/run_balance.py
"""
import statistics as stt
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc

cc.N_AGENTS = 3
cc.GRID = 10
import _marl_core as core          # noqa: E402
import _frontier_core as fc        # noqa: E402
from marl_frontier_comm import FrontierCommAttnAC as M  # noqa: E402

STEPS, N, COMM = 55, 3, 5
CELLS = 100
cc.MAX_STEPS = STEPS
core.T = STEPS
fc.T = STEPS
core.ITERS = 200
ENV = dict(grid_h=10, grid_w=10, n_agents=N, comm_r=COMM, spawn_radius=2)
SHAPE = dict(w_shape1=1.0, w_shape3=2.0)
SEEDS = [1, 2]
W_DEGREE_SWEEP = [0.0, 0.1, 0.3, 1.0]   # OFF -> FULL connectivity pressure


def evalp(env, vdir):
    m = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"), M(env, jax.random.PRNGKey(0)))
    tr = jax.vmap(lambda k: fc._rollout_masked(env, m, STEPS, k))(
        jax.random.split(jax.random.PRNGKey(7), 32))
    st = tr["world"]
    cov = float((st.team_explored[:, -1].reshape(32, -1).sum(-1) / CELLS).mean())
    d = jnp.max(jnp.abs(st.pos[..., :, None, :] - st.pos[..., None, :, :]), -1)
    reach = jax.vmap(jax.vmap(jax.vmap(cc._reach)))(d <= COMM)
    giant = reach.sum(-1).max(-1)
    conn_frac = float((giant == N).mean())
    maxd = float(jnp.where(~jnp.eye(N, dtype=bool), d[:, -1], 0).max((-1, -2)).mean())
    return cov, conn_frac, maxd


base = core.new_run_dir()
print(f"BALANCE curve · 3 agents / 10x10 / comm_r={COMM} / {STEPS} steps · {base}\n", flush=True)
print(f"{'w_degree':>9}{'coverage':>11}{'conn-frac':>11}{'max-dist':>10}", flush=True)
for wd in W_DEGREE_SWEEP:
    res = []
    for sd in SEEDS:
        env = cc.make_env(w_cov=1.0, w_degree=wd, **SHAPE, **ENV)
        b = fc.run(env, M, f"wd{wd}-s{sd}", outdir=base, seed=sd,
                   target_cov=None, patience=999, min_iters=999, mask_collisions=False)
        res.append(evalp(env, Path(b["dir"])))
    cov = stt.mean([r[0] for r in res])
    cf = stt.mean([r[1] for r in res])
    md = stt.mean([r[2] for r in res])
    print(f"  ===> {wd:>5.2f}{cov*100:>9.1f}%{cf*100:>9.0f}%{md:>10.1f}", flush=True)
print(f"\n  read the curve: more coverage as connectivity drops. pick the knee (e.g. ~85% conn).")
print(f"artifacts: {base}", flush=True)
