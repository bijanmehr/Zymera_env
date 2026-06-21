"""Test the dense PBRS shaping (Φ1+Φ3) on the two-head model under MASKED hard
collision. Does the dense signal fix the sparse-coverage collapse, and does the
formation head then hold the swarm together?  Dev world (10×10).

    python examples/run_shaping.py
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
from marl_frontier_comm import FrontierCommAttnAC

core.ITERS = 150
N = cc.N_AGENTS
SHAPE = dict(w_cov=1.0, w_shape1=1.0, w_shape3=2.0)
CONFIGS = [
    ("shaping (no formation)", SHAPE),
    ("shaping + formation",    dict(**SHAPE, w_cohesion=2.0, cohesion_leash=2.0)),
]


def evaluate(env, vdir):
    m = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"), FrontierCommAttnAC(env, jax.random.PRNGKey(0)))
    tr = jax.vmap(lambda k: fc._rollout_masked(env, m, cc.MAX_STEPS, k))(jax.random.split(jax.random.PRNGKey(7), 32))
    st = tr["world"]
    cov = float((st.team_explored[:, -1].reshape(32, -1).sum(-1) / (cc.GRID ** 2)).mean())
    redun = float((st.explored_by[:, -1].sum((-3, -2, -1)) / jnp.maximum(st.team_explored[:, -1].sum((-1, -2)), 1)).mean())
    d = jnp.max(jnp.abs(st.pos[..., :, None, :] - st.pos[..., None, :, :]), -1)
    off = ~jnp.eye(N, dtype=bool)
    reach = jax.vmap(jax.vmap(cc._reach))(d <= cc.COMM_R)
    giant = float((reach.sum(-1).max(-1) / N).mean())
    maxd = float(jnp.where(off, d, 0).max((-1, -2)).mean())
    formed = float((jnp.where(off, d, jnp.inf).min(-1) <= 2).mean())
    coll = float(((((d == 0) & off).sum((-1, -2)) / 2.0)).mean())
    return cov, giant, maxd, formed, redun, coll


base = core.new_run_dir()
print(f"PBRS shaping + two-head + masked collision (dev {cc.GRID}×{cc.GRID}) → {base}\n")
rows = []
for name, rw in CONFIGS:
    env = cc.make_env(**rw)
    best = fc.run(env, FrontierCommAttnAC, name, outdir=base, seed=1,
                  target_cov=None, patience=999, min_iters=999, mask_collisions=True)
    rows.append((name, evaluate(env, Path(best["dir"]))))

print("\n========= shaping test · eval (32 eps) =========")
print(f"{'config':24s} {'coverage':>9} {'giant':>7} {'max-dist':>9} {'formed':>7} {'redun':>7} {'coll':>6}")
for name, (c, g, m, f, r, cl) in rows:
    print(f"{name:24s} {c*100:8.1f}% {g*100:6.0f}% {m:9.1f} {f*100:6.0f}% {r:6.2f}× {cl:6.2f}")
print(f"artifacts: {base}")
