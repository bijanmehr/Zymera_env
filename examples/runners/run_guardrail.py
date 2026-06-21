"""HARD connectivity guardrail (the comm-graph-integrity version) vs the soft
degree-floor vs no constraint. The guardrail masks any move that would disconnect
the comm graph -- integrity GUARANTEED by construction, every step -- while letting
agents spread right up to the edge (no over-cautious soft penalty).

3 agents / 10x10 / comm_r=3 (tight: connectivity actually binds) / 55 steps.
  guardrail   : no soft penalty, hard mask  -> expect 100% connected + coverage >= soft
  degree-floor: soft penalty (incentive)    -> ~60% / 97% connected
  none        : free spread                 -> high coverage, but connectivity drops

    python -u examples/run_guardrail.py
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

STEPS, N, COMM = 55, 3, 3
CELLS = 100
cc.MAX_STEPS = STEPS
core.T = STEPS
fc.T = STEPS
core.ITERS = 200
ENV = dict(grid_h=10, grid_w=10, n_agents=N, comm_r=COMM, spawn_radius=2)
SHAPE = dict(w_shape1=1.0, w_shape3=2.0)
SEEDS = [1, 2]


def evalp(env, vdir, guard):
    m = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"), M(env, jax.random.PRNGKey(0)))
    cr = int(env.comm_r) if guard else None                       # eval with the same guardrail
    tr = jax.vmap(lambda k: fc._rollout_masked(env, m, STEPS, k, cr))(
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
print(f"GUARDRAIL vs soft vs none · 3 agents / 10x10 / comm_r={COMM} / {STEPS} steps · {base}\n", flush=True)
# (name, env connectivity kwargs, guard_conn)
VARIANTS = [
    ("guardrail", dict(), True),
    ("degree-floor", dict(w_degree=1.0), False),
    ("none", dict(), False),
]
for name, ckw, guard in VARIANTS:
    res = []
    for sd in SEEDS:
        env = cc.make_env(w_cov=1.0, **SHAPE, **ckw, **ENV)
        b = fc.run(env, M, f"{name}-s{sd}", outdir=base, seed=sd,
                   target_cov=None, patience=999, min_iters=999,
                   mask_collisions=False, guard_conn=guard)
        res.append(evalp(env, Path(b["dir"]), guard))
    covs = [r[0] for r in res]
    print(f"  ===> {name:14s} cov {stt.mean(covs)*100:5.1f}% ±{stt.pstdev(covs)*100:3.1f}  "
          f"conn-frac {stt.mean([r[1] for r in res])*100:3.0f}%  "
          f"maxd {stt.mean([r[2] for r in res]):.1f}  seeds={[round(c*100, 1) for c in covs]}", flush=True)
print(f"\n  guardrail wins if conn-frac=100% (guaranteed) AND coverage beats the soft degree-floor.")
print(f"artifacts: {base}", flush=True)
