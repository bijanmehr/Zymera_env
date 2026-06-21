"""Smooth LOCAL connectivity BARRIER vs the binary degree-floor. Each agent gets a
penalty that is ~0 while its nearest teammate is comfortably in range and ramps
(exp, capped) as that distance nears comm_r -- a gradient-bearing 'don't drift out
of contact' signal, vs the degree-floor's flat 'you're isolated' penalty.

Dev world 10x10 / 3 agents / comm_r=3 (tight, so connectivity is a real constraint) /
55 steps. Pure cover + connectivity (no collision masking, to isolate the question).
Watching: coverage, giant-component, and CONNECTED-FRACTION (steps fully connected --
the stability the smooth gradient should buy).

    python -u examples/run_cbf_barrier.py
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

cc.N_AGENTS = 3
cc.GRID = 10
import _marl_core as core          # noqa: E402
import _frontier_core as fc        # noqa: E402
from marl_frontier_comm import FrontierCommAttnAC as M  # noqa: E402

STEPS, N, COMM = 55, 3, 5    # comm_r=5: real ramp room (dists 1..5) for a SMOOTH barrier
CELLS = 100
cc.MAX_STEPS = STEPS
core.T = STEPS
fc.T = STEPS
core.ITERS = 200
ENV = dict(grid_h=10, grid_w=10, n_agents=N, comm_r=COMM, spawn_radius=2)
SHAPE = dict(w_shape1=1.0, w_shape3=2.0)
SEEDS = [1, 2, 3]


def evalp(env, vdir):
    m = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"), M(env, jax.random.PRNGKey(0)))
    tr = jax.vmap(lambda k: fc._rollout_masked(env, m, STEPS, k))(
        jax.random.split(jax.random.PRNGKey(7), 32))
    st = tr["world"]
    cov = float((st.team_explored[:, -1].reshape(32, -1).sum(-1) / CELLS).mean())
    d = jnp.max(jnp.abs(st.pos[..., :, None, :] - st.pos[..., None, :, :]), -1)   # (32,T+1,N,N)
    reach = jax.vmap(jax.vmap(jax.vmap(cc._reach)))(d <= COMM)                     # (32,T+1,N,N)
    giant = reach.sum(-1).max(-1)                                                  # (32,T+1) largest comp
    giant_final = float((giant[:, -1] / N).mean())
    conn_frac = float((giant == N).mean())                                        # frac of steps fully connected
    off = ~jnp.eye(N, dtype=bool)
    maxd = float(jnp.where(off, d[:, -1], 0).max((-1, -2)).mean())
    return cov, giant_final, conn_frac, maxd


base = core.new_run_dir()
print(f"CBF BARRIER vs degree-floor · 3 agents / 10x10 / comm_r={COMM} / {STEPS} steps · {base}\n", flush=True)
VARIANTS = [
    ("degree-floor", dict(w_degree=1.0)),
    ("cbf-barrier", dict(w_conn_barrier=0.1, barrier_safe=3.5, barrier_sharp=1.5)),  # bite only near comm_r=5
]
for name, ckw in VARIANTS:
    res = []
    for sd in SEEDS:
        env = cc.make_env(w_cov=1.0, **SHAPE, **ckw, **ENV)
        b = fc.run(env, M, f"{name}-s{sd}", outdir=base, seed=sd,
                   target_cov=None, patience=999, min_iters=999, mask_collisions=False)
        res.append(evalp(env, Path(b["dir"])))
    covs = [r[0] for r in res]
    print(f"  ===> {name:14s} cov {stt.mean(covs)*100:5.1f}% ±{stt.pstdev(covs)*100:3.1f}  "
          f"giant {stt.mean([r[1] for r in res])*100:3.0f}%  conn-frac {stt.mean([r[2] for r in res])*100:3.0f}%  "
          f"maxd {stt.mean([r[3] for r in res]):.1f}  seeds={[round(c*100, 1) for c in covs]}", flush=True)
print(f"\n  smooth barrier should match-or-beat coverage AND hold connectivity more stably (higher conn-frac).")
print(f"artifacts: {base}", flush=True)
