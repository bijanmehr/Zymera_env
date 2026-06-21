"""Curriculum for 1×1 visit-coverage + hard collision + formation, two-head model.

  phase 1 (warm-up): two-head + coverage + PBRS shaping, NO collision constraint
                     → learn to spread & cover (escape the clump).
  phase 2 (fine-tune): warm-start phase-1, turn on MASKED hard collision AND the
                     capped giant-component formation reward → hold coverage, keep
                     the swarm connected, 0 collisions.

Dev world (10×10); everything size-invariant so the policy transfers to 16×16.

    python examples/run_curriculum.py
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

N = cc.N_AGENTS
SHAPE = dict(w_shape1=1.0, w_shape3=2.0)
P1_ITERS, P2_ITERS = 120, 100


def graded_eval(env, vdir):
    m = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"), FrontierCommAttnAC(env, jax.random.PRNGKey(0)))
    tr = jax.vmap(lambda k: fc._rollout_masked(env, m, cc.MAX_STEPS, k))(jax.random.split(jax.random.PRNGKey(7), 32))
    st = tr["world"]
    cov = float((st.team_explored[:, -1].reshape(32, -1).sum(-1) / (cc.GRID ** 2)).mean())
    d = jnp.max(jnp.abs(st.pos[..., :, None, :] - st.pos[..., None, :, :]), -1)
    off = ~jnp.eye(N, dtype=bool)
    reach = jax.vmap(jax.vmap(cc._reach))(d <= cc.COMM_R)
    giant = float((reach.sum(-1).max(-1) / N).mean())
    maxd = float(jnp.where(off, d, 0).max((-1, -2)).mean())
    coll = float(((((d == 0) & off).sum((-1, -2)) / 2.0)).mean())
    return cov, giant, maxd, coll


base = core.new_run_dir()
print(f"CURRICULUM (two-head, 1×1 cover + masked collision + formation) → {base}\n")

# ---- phase 1: coverage + shaping, no collision constraint ----
core.ITERS = P1_ITERS
env1 = cc.make_env(w_cov=1.0, **SHAPE)
b1 = fc.run(env1, FrontierCommAttnAC, "phase1-spread", outdir=base, seed=1,
            target_cov=None, patience=999, min_iters=999, mask_collisions=False)
m1 = eqx.tree_deserialise_leaves(str(Path(b1["dir"]) / "checkpoint.eqx"),
                                 FrontierCommAttnAC(env1, jax.random.PRNGKey(0)))
print(f"--- phase1 (spread): {b1['cov']*100:.1f}% — warm-starting phase2 (masked + formation) ---\n")

# ---- phase 2: warm-start; masked hard collision + capped giant-component formation ----
core.ITERS = P2_ITERS
env2 = cc.make_env(w_cov=1.0, w_conn=1.5, conn_cap=N - 1, **SHAPE)
b2 = fc.run(env2, FrontierCommAttnAC, "phase2-form", outdir=base, seed=2, init_model=m1,
            target_cov=None, patience=999, min_iters=999, mask_collisions=True)

cov, giant, maxd, coll = graded_eval(env2, Path(b2["dir"]))
print("\n========= curriculum result · eval (32 eps, masked) =========")
print(f"coverage {cov*100:.1f}%   giant-comp {giant*100:.0f}%   max-dist {maxd:.1f}   collisions {coll:.2f}")
print(f"(phase1 spread {b1['cov']*100:.0f}% → phase2 with masking+formation)")
print(f"artifacts: {base}")
