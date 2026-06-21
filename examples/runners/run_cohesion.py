"""Cohesion-leash experiment: local freedom but no agent leaves the herd.
Two-head model (FrontierCommAttnAC) + reward = coverage + cohesion leash (soft
pull-back when nearest teammate > leash) + hard collision masking. No CBF, no
giant-component reward — the leash alone tethers every agent.

    python examples/run_cohesion.py
"""
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc
import _marl_core as core
import _frontier_core as fc
import zymera
from marl_frontier_comm import FrontierCommAttnAC
from marl_frontier_attn import FrontierAttnAC

core.ITERS = 160
N = cc.N_AGENTS
# what WORKED (capped giant reward → spread+recover to 89.5%/87%) + a milder leash
# (=comm range) that bounds the roamer so no agent fully leaves. Hard collision.
REWARD = dict(w_cov=1.0, w_conn=4.0, conn_cap=3, w_coll=0.0, w_overlap=0.0,
              w_cohesion=2.0, cohesion_leash=5.0, hard_collision=True)
CONFIGS = [("single-head", FrontierAttnAC), ("two-head", FrontierCommAttnAC)]


def evaluate(env, vdir, Model):
    m = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"), Model(env, jax.random.PRNGKey(0)))
    pol = lambda o, k: jax.random.categorical(k, m.logits(o), -1).astype(jnp.int32)
    tr = jax.vmap(lambda k: zymera.rollout(env, pol, cc.MAX_STEPS, k))(jax.random.split(jax.random.PRNGKey(7), 32))
    st = tr["world"]
    cov = float((st.team_explored[:, -1].reshape(32, -1).sum(-1) / (cc.GRID ** 2)).mean())
    d = jnp.max(jnp.abs(st.pos[..., :, None, :] - st.pos[..., None, :, :]), -1)   # (32,T+1,N,N)
    off = ~jnp.eye(N, dtype=bool)
    reach = jax.vmap(jax.vmap(cc._reach))(d <= cc.COMM_R)
    giant = float((reach.sum(-1).max(-1) / N).mean())
    maxd = float(jnp.where(off, d, 0).max((-1, -2)).mean())
    nn = jnp.where(off, d, jnp.inf).min(-1)                                        # (32,T+1,N) nearest-nbr
    tethered = float((nn <= cc.COMM_R).mean())          # fraction of (agent,step) with a teammate in range
    return cov, giant, maxd, tethered


base = core.new_run_dir()
print(f"cohesion-leash (no agent leaves the herd) → {base}\n")
rows = []
for name, Model in CONFIGS:
    env = cc.make_env(**REWARD)
    best = fc.run(env, Model, name, outdir=base, target_cov=None, patience=70, min_iters=110)
    rows.append((name, evaluate(env, Path(best["dir"]), Model)))

print("\n========= cohesion-leash · eval (32 eps) =========")
print(f"{'config':22s} {'coverage':>9} {'giant-comp':>11} {'max-dist':>9} {'tethered':>9}")
for name, (c, g, m, t) in rows:
    print(f"{name:22s} {c*100:8.1f}% {g*100:10.0f}% {m:9.1f} {t*100:8.0f}%")
print("(tethered = % of agent-steps with a teammate in comm range; high = nobody leaves)")
print(f"artifacts: {base}")
