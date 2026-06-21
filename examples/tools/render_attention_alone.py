"""GIF + HTML report for the attention-alone policy (FrontierCommAttnAC, coverage
reward only) so it can be audited visually — the comm-graph overlay shows the
connectivity/huddle behaviour. Renders a representative episode (coverage nearest
the 32-episode mean), sampling from π (eps=0)."""
import sys
import types
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc
import zymera
from zymera import viz
from marl_frontier_comm import FrontierCommAttnAC

VDIR = Path("experiments/marl/20260611-015906/comm-noincentive")
env = cc.make_env()
model = eqx.tree_deserialise_leaves(str(VDIR / "checkpoint.eqx"),
                                    FrontierCommAttnAC(env, jax.random.PRNGKey(0)))

def policy(o, k):
    return jax.random.categorical(k, model.logits(o), -1).astype(jnp.int32)

trajs = jax.vmap(lambda k: zymera.rollout(env, policy, cc.MAX_STEPS, k))(
    jax.random.split(jax.random.PRNGKey(7), 32))
st = trajs["world"]
cov = st.team_explored[:, -1].reshape(32, -1).sum(-1) / (cc.GRID ** 2)
rep = int(np.argmin(np.abs(np.asarray(cov) - float(cov.mean()))))
ep = jax.tree_util.tree_map(lambda x: x[rep], st)
n = int(ep.pos.shape[0])

def snap(t):
    s = jax.tree_util.tree_map(lambda x: x[t], ep)
    return types.SimpleNamespace(
        visited=np.asarray(s.team_explored),
        explored=np.asarray(s.explored_by).sum(0),
        wall=np.asarray(s.wall),
        body=types.SimpleNamespace(position=np.asarray(s.pos)))

viz_traj = {"world": [snap(t) for t in range(n)]}
out = VDIR / "report.html"
viz.make_report(viz_traj, str(out),
                title=f"attention-alone (no incentive) · {float(cov.mean())*100:.0f}% cov · "
                      f"84% giant-component · colliding huddle",
                fps=8, comm_radius=cc.COMM_R)
print(f"representative episode #{rep} (cov {float(cov[rep])*100:.1f}%)")
print(f"wrote {out}")
