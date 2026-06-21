"""Two CBF studies on the frontier model (FrontierAttnAC), into one run dir:

  cbf-collision-only : reward = coverage + collision-CBF ONLY (w_cbf_conn=0).
       Isolates the LOCAL pairwise barrier — does it keep collisions ~0 while
       coverage drives? (Prediction: yes; connectivity disperses, no conn incentive.)

  cbf-conn-strong    : coverage + collision-CBF + STRONG/SHARP connectivity-CBF
       (w_cbf_conn=40, cbf_sharp=8 so λ₂≥ε ≈ binary-connected, fixing the permissive
       sigmoid tails). Tests whether the connectivity barrier can finally hold the chain.

    python examples/run_cbf_studies.py
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
CONFIGS = [
    ("cbf-collision-only", dict(w_cov=1.0, w_conn=0.0, w_coll=0.0, w_overlap=0.0,
                                w_cbf_conn=0.0, w_cbf_coll=8.0, cbf_alpha=1.0)),
    ("cbf-conn-strong",    dict(w_cov=1.0, w_conn=0.0, w_coll=0.0, w_overlap=0.0,
                                w_cbf_conn=40.0, w_cbf_coll=8.0, cbf_alpha=1.0,
                                cbf_eps=0.3, cbf_sharp=8.0)),
]


def evaluate(env, vdir):
    like = FrontierAttnAC(env, jax.random.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"), like)
    def policy(o, k):
        return jax.random.categorical(k, model.logits(o), -1).astype(jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(7), 32)
    tr = jax.vmap(lambda k: zymera.rollout(env, policy, cc.MAX_STEPS, k))(keys)
    st = tr["world"]
    cov = st.team_explored[:, -1].reshape(32, -1).sum(-1) / (cc.GRID ** 2)
    pos = st.pos
    d = jnp.max(jnp.abs(pos[..., :, None, :] - pos[..., None, :, :]), -1)
    off = ~jnp.eye(cc.N_AGENTS, dtype=bool)
    conn = jax.vmap(jax.vmap(lambda dd: cc._connected(dd <= cc.COMM_R)))(d).mean(-1)
    coll = (((d == 0) & off).sum((-1, -2)) / 2.0).mean(-1)
    return float(cov.mean()), float(conn.mean()), float(coll.mean())


base = core.new_run_dir()
print(f"CBF studies → {base}\n")
rows = []
for name, rw in CONFIGS:
    env = cc.make_env(**rw)
    best = fc.run(env, FrontierAttnAC, name, outdir=base,
                  target_cov=None, patience=70, min_iters=110)
    rows.append((name, evaluate(env, Path(best["dir"]))))

print("\n============ CBF studies · eval (32 eps) ============")
print(f"{'config':22s} {'coverage':>9} {'connected':>10} {'collisions':>11}")
for name, (c, k, cl) in rows:
    print(f"{name:22s} {c*100:8.1f}% {k*100:9.0f}% {cl:11.2f}")
print(f"artifacts: {base}")
