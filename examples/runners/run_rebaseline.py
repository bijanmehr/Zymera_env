"""Re-baseline on the CORRECTED env (explore 1×1 / scan 3×3, 70 steps, true
visit-coverage). Two configs on the frontier-attention model:

  coverage-only : pure coverage + hard collision → the new coverage CEILING.
  cap3+leash5   : coverage + capped giant-component (cap=3) + cohesion leash (=comm
                  range) + hard collision → the goal (coverage + connectivity + no
                  agent leaves the herd + no collisions).

Judged on coverage, giant-component, max-dist, %tethered, and REDUNDANCY (efficiency).

    python examples/run_rebaseline.py
"""
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc
import _marl_core as core
import _frontier_core as fc
import zymera
from marl_frontier_attn import FrontierAttnAC

core.ITERS = 160
N = cc.N_AGENTS
CONFIGS = [
    ("coverage-only", dict(w_cov=1.0, hard_collision=True)),
    ("cap+leash",     dict(w_cov=1.0, w_conn=4.0, conn_cap=N - 1, w_cohesion=2.0,
                           cohesion_leash=float(cc.COMM_R), hard_collision=True)),
]


def evaluate(env, vdir):
    m = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"), FrontierAttnAC(env, jax.random.PRNGKey(0)))
    pol = lambda o, k: jax.random.categorical(k, m.logits(o), -1).astype(jnp.int32)
    tr = jax.vmap(lambda k: zymera.rollout(env, pol, cc.MAX_STEPS, k))(jax.random.split(jax.random.PRNGKey(7), 32))
    st = tr["world"]
    cov = float((st.team_explored[:, -1].reshape(32, -1).sum(-1) / (cc.GRID ** 2)).mean())
    redun = float((st.explored_by[:, -1].sum((-3, -2, -1)) / jnp.maximum(st.team_explored[:, -1].sum((-1, -2)), 1)).mean())
    d = jnp.max(jnp.abs(st.pos[..., :, None, :] - st.pos[..., None, :, :]), -1)
    off = ~jnp.eye(N, dtype=bool)
    reach = jax.vmap(jax.vmap(cc._reach))(d <= cc.COMM_R)
    giant = float((reach.sum(-1).max(-1) / N).mean())
    maxd = float(jnp.where(off, d, 0).max((-1, -2)).mean())
    tethered = float((jnp.where(off, d, jnp.inf).min(-1) <= cc.COMM_R).mean())
    return cov, giant, maxd, tethered, redun


base = core.new_run_dir()
print(f"RE-BASELINE on corrected env (explore 1×1, scan 3×3) → {base}\n")
rows = []
for name, rw in CONFIGS:
    env = cc.make_env(**rw)
    best = fc.run(env, FrontierAttnAC, name, outdir=base, target_cov=None, patience=70, min_iters=110)
    rows.append((name, evaluate(env, Path(best["dir"]))))

print("\n=========== RE-BASELINE · eval (32 eps, visit-coverage) ===========")
print(f"{'config':14s} {'coverage':>9} {'giant':>7} {'max-dist':>9} {'tethered':>9} {'redundancy':>11}")
for name, (c, g, m, t, r) in rows:
    print(f"{name:14s} {c*100:8.1f}% {g*100:6.0f}% {m:9.1f} {t*100:8.0f}% {r:10.2f}×")
print(f"artifacts: {base}")
