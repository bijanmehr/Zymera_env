"""3-way leg: CBF + inter-agent attention. FrontierCommAttnAC (neighbour-attention
head) trained with the SAME soft-CBF reward as the CBF-alone run, so the only
difference vs CBF-alone is the attention head. Tests whether coordination capability
helps the CBF incentive hold the connected formation.

    python examples/run_cbf_attn.py
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

core.ITERS = 160
REWARD = dict(w_cov=1.0, w_conn=0.0, w_coll=0.0, w_overlap=0.0,
              w_cbf_conn=15.0, w_cbf_coll=8.0, cbf_alpha=1.0, cbf_eps=0.3)

env = cc.make_env(**REWARD)
best = fc.run(env, FrontierCommAttnAC, "cbf+attention", seed=1,
              target_cov=None, patience=70, min_iters=110)
vdir = Path(best["dir"])

like = FrontierCommAttnAC(env, jax.random.PRNGKey(0))
model = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"), like)

def policy(obs, key):
    return jax.random.categorical(key, model.logits(obs), axis=-1).astype(jnp.int32)

keys = jax.random.split(jax.random.PRNGKey(7), 32)
tr = jax.vmap(lambda k: zymera.rollout(env, policy, cc.MAX_STEPS, k))(keys)
st = tr["world"]
cov = st.team_explored[:, -1].reshape(32, -1).sum(-1) / (cc.GRID ** 2)
pos = st.pos
d = jnp.max(jnp.abs(pos[..., :, None, :] - pos[..., None, :, :]), -1)
off = ~jnp.eye(cc.N_AGENTS, dtype=bool)
conn = jax.vmap(jax.vmap(lambda dd: cc._connected(dd <= cc.COMM_R)))(d).mean(-1)
coll = (((d == 0) & off).sum((-1, -2)) / 2.0).mean(-1)
maxd = jnp.where(off, d, 0).max((-1, -2))[:, -1]

print("\n========= CBF + attention · eval (32 eps) =========")
print(f"coverage {float(cov.mean())*100:.1f}%   connected {float(conn.mean())*100:.0f}%   "
      f"max-dist {float(maxd.mean()):.1f}   collisions {float(coll.mean()):.2f}")
print(f"artifacts: {vdir}")
