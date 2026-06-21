"""Two specialized attention heads: head-1 uncertainty-reduction (coverage),
head-2 formation-keeping (stay together). FrontierCommAttnAC trained with
coverage + cohesion-leash (formation). Tested both without collision handling
and with masked hard-collision, on the fast dev world.

    python examples/run_twohead_formation.py
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
from marl_frontier_comm import FrontierCommAttnAC

core.ITERS = 130
N = cc.N_AGENTS
REWARD = dict(w_cov=1.0, w_cohesion=2.0, cohesion_leash=2.0)   # coverage + formation-keeping
CONFIGS = [("twohead-formation NOcoll", False), ("twohead-formation MASKED", True)]


def evaluate(env, vdir, mask):
    m = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"), FrontierCommAttnAC(env, jax.random.PRNGKey(0)))
    pol = lambda o, k: jax.random.categorical(k, m.logits(o), -1).astype(jnp.int32)
    if mask:
        rollout = lambda k: fc._rollout_masked(env, m, cc.MAX_STEPS, k)
    else:
        rollout = lambda k: zymera.rollout(env, pol, cc.MAX_STEPS, k)
    tr = jax.vmap(rollout)(jax.random.split(jax.random.PRNGKey(7), 32))
    st = tr["world"]
    cov = float((st.team_explored[:, -1].reshape(32, -1).sum(-1) / (cc.GRID ** 2)).mean())
    d = jnp.max(jnp.abs(st.pos[..., :, None, :] - st.pos[..., None, :, :]), -1)
    off = ~jnp.eye(N, dtype=bool)
    reach = jax.vmap(jax.vmap(cc._reach))(d <= cc.COMM_R)
    giant = float((reach.sum(-1).max(-1) / N).mean())
    maxd = float(jnp.where(off, d, 0).max((-1, -2)).mean())
    nn = jnp.where(off, d, jnp.inf).min(-1)
    formed = float((nn <= 2).mean())                 # within formation leash of a teammate
    coll = float(((((d == 0) & off).sum((-1, -2)) / 2.0)).mean())
    return cov, giant, maxd, formed, coll


base = core.new_run_dir()
print(f"two-head (uncertainty + formation) → {base}\n")
rows = []
for name, mask in CONFIGS:
    env = cc.make_env(**REWARD)
    best = fc.run(env, FrontierCommAttnAC, name, outdir=base, seed=1,
                  target_cov=None, patience=999, min_iters=999, mask_collisions=mask)
    rows.append((name, evaluate(env, Path(best["dir"]), mask)))

print("\n========= two-head uncertainty+formation · eval (32 eps) =========")
print(f"{'config':28s} {'coverage':>9} {'giant':>7} {'max-dist':>9} {'in-formation':>13} {'collisions':>11}")
for name, (c, g, m, f, cl) in rows:
    print(f"{name:28s} {c*100:8.1f}% {g*100:6.0f}% {m:9.1f} {f*100:12.0f}% {cl:11.2f}")
print(f"artifacts: {base}")
