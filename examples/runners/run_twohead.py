"""Does the second (neighbour) attention head help, under a balanced reward and NO CBF?

Head-to-head, identical reward = coverage + GRADED connectivity (reach-1)/(N-1) + collision
penalty (no CBF, no overlap). Only difference = the architecture:
  uncertainty-only : FrontierAttnAC      (1 head: spatial frontier attention)
  two-head         : FrontierCommAttnAC  (+ inter-agent neighbour attention)
Idea: the heads supply capability, the reward supplies the coverage/connectivity balance,
and the agents learn to balance. Measured by GIANT-COMPONENT (relaxed connectivity).

    python examples/run_twohead.py
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
from marl_frontier_comm import FrontierCommAttnAC

core.ITERS = 160
# balance = coverage + CAPPED giant-component connectivity (cap=3 ⇒ "3 connected +
# 1 roamer" scores like a full clump → no clump incentive, roamer free to cover);
# collisions handled the HARD way (env masks colliding moves), so w_coll=0. No CBF.
REWARD = dict(w_cov=1.0, w_conn=4.0, conn_cap=3, w_coll=0.0, w_overlap=0.0, hard_collision=True)
N = cc.N_AGENTS
CONFIGS = [("uncertainty-only", FrontierAttnAC), ("two-head", FrontierCommAttnAC)]


def graded_eval(env, vdir, Model):
    m = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"), Model(env, jax.random.PRNGKey(0)))
    pol = lambda o, k: jax.random.categorical(k, m.logits(o), -1).astype(jnp.int32)
    tr = jax.vmap(lambda k: zymera.rollout(env, pol, cc.MAX_STEPS, k))(
        jax.random.split(jax.random.PRNGKey(7), 32))
    st = tr["world"]
    cov = float((st.team_explored[:, -1].reshape(32, -1).sum(-1) / (cc.GRID ** 2)).mean())
    d = jnp.max(jnp.abs(st.pos[..., :, None, :] - st.pos[..., None, :, :]), -1)
    reach = jax.vmap(jax.vmap(cc._reach))(d <= cc.COMM_R)
    giant = float((reach.sum(-1).max(-1) / N).mean())
    off = ~jnp.eye(N, dtype=bool)
    pairs = float(((reach & off).sum((-1, -2)) / (N * (N - 1))).mean())
    coll = float(((((d == 0) & off).sum((-1, -2)) / 2.0)).mean())
    return cov, giant, pairs, coll


base = core.new_run_dir()
print(f"two-head vs single-head (balanced reward, no CBF) → {base}\n")
rows = []
for name, Model in CONFIGS:
    env = cc.make_env(**REWARD)
    best = fc.run(env, Model, name, outdir=base, target_cov=None, patience=70, min_iters=110)
    rows.append((name, graded_eval(env, Path(best["dir"]), Model)))

print("\n========= two-head vs single-head · eval (32 eps) =========")
print(f"{'config':18s} {'coverage':>9} {'giant-comp':>11} {'pairs':>7} {'collisions':>11}")
for name, (c, g, p, cl) in rows:
    print(f"{name:18s} {c*100:8.1f}% {g*100:10.0f}% {p*100:6.0f}% {cl:11.2f}")
print(f"artifacts: {base}")
