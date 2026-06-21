"""Generate the self-contained HTML report for the CTDE variant.

Pipeline:
  1. train CTDE via the shared core (writes metrics.npz / checkpoint.eqx /
     config.json into experiments/marl/<timestamp>/ctde/) — OR, if a run dir
     with a saved checkpoint is passed on argv, reload it and skip training;
  2. EVAL over many episodes by SAMPLING from π (ε=0), not argmax — argmax
     collapses the symmetry-breaking the policy uses for who-goes-where;
  3. report the mean eval coverage/connectivity/spread over those episodes, and
     embed the REPRESENTATIVE episode (final coverage closest to the mean) so
     the gif is neither cherry-picked nor an unlucky tail;
  4. render report.html (episode gif WITH comm-graph edges so the stretched
     relay chain is visible, coverage-over-time, team visit heatmap, training
     curves) next to the checkpoint.

    python examples/report_ctde.py                       # train fresh, then report
    python examples/report_ctde.py experiments/marl/<ts> # reuse saved checkpoint
"""

import sys
import types
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc
import _marl_core as core
import zymera
from zymera import viz
from marl_ctde import CTDE_AC

N_EVAL = 16   # eval episodes (the report number is their mean, not one draw)

env = cc.make_env()
like = CTDE_AC(env, jax.random.PRNGKey(0))                     # structure to fill

# ---- 1. train, or reload a saved checkpoint --------------------------------
arg = sys.argv[1] if len(sys.argv) > 1 else None
ckpt = Path(arg) / "ctde" / "checkpoint.eqx" if arg else None
if ckpt and ckpt.exists():
    vdir = ckpt.parent
    model = eqx.tree_deserialise_leaves(str(ckpt), like)
    print(f"reloaded checkpoint → {ckpt}")
else:
    best = core.run(env, CTDE_AC, "CTDE", seed=1)
    vdir = Path(best["dir"])
    model = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"), like)

# ---- 2. eval over N_EVAL episodes (sample from π, ε=0) ----------------------
def policy(obs, key):
    return jax.random.categorical(key, model.logits(obs), axis=-1).astype(jnp.int32)

keys = jax.random.split(jax.random.PRNGKey(7), N_EVAL)
trajs = jax.vmap(lambda k: zymera.rollout(env, policy, cc.MAX_STEPS, k))(keys)
states = trajs["world"]                                        # leading (N_EVAL, T+1, ...)

cov_ep = states.team_explored[:, -1].reshape(N_EVAL, -1).sum(-1) / (cc.GRID ** 2)  # (E,)
pos_all = states.pos                                           # (E, T+1, N, 2)
d_all = jnp.max(jnp.abs(pos_all[..., :, None, :] - pos_all[..., None, :, :]), -1)  # (E,T+1,N,N)
off = ~jnp.eye(cc.N_AGENTS, dtype=bool)
maxd_ep = jnp.where(off, d_all, 0).max((-1, -2))[:, -1]        # final-step spread (E,)
conn_all = jax.vmap(jax.vmap(lambda dd: cc._connected(dd <= cc.COMM_R)))(d_all)    # (E,T+1)

cov_ep, maxd_ep = np.asarray(cov_ep), np.asarray(maxd_ep)
eval_cov, eval_cov_sd = float(cov_ep.mean()), float(cov_ep.std())
eval_conn = float(np.asarray(conn_all).mean())
eval_maxd = float(maxd_ep.mean())
rep = int(np.argmin(np.abs(cov_ep - cov_ep.mean())))          # representative episode
print(f"eval over {N_EVAL} episodes (ε=0): coverage {eval_cov*100:.1f}% ± {eval_cov_sd*100:.1f}  "
      f"connected {eval_conn*100:.0f}%  max-dist {eval_maxd:.1f}")
print(f"embedding representative episode #{rep} (final coverage {cov_ep[rep]*100:.1f}%)")

# ---- 3. adapt the representative episode → viz world snapshots --------------
ep = jax.tree_util.tree_map(lambda x: x[rep], states)         # one episode (T+1, ...)
n_frames = int(ep.pos.shape[0])

def snap(t):
    st = jax.tree_util.tree_map(lambda x: x[t], ep)
    return types.SimpleNamespace(
        visited=np.asarray(st.team_explored),                       # gif background
        explored=np.asarray(st.explored_by).sum(0),                 # per-cell agent count → redundancy heatmap
        wall=np.asarray(st.wall),
        body=types.SimpleNamespace(position=np.asarray(st.pos)),
    )

viz_traj = {"world": [snap(t) for t in range(n_frames)]}

# ---- 4. training curves + render -------------------------------------------
m = np.load(vdir / "metrics.npz")
metrics = {
    "train coverage %": m["cov"] * 100,
    "connected %": m["conn"] * 100,
    "max pairwise dist": m["maxd"],
}
out = vdir / "report.html"
viz.make_report(
    viz_traj, str(out),
    title=(f"CTDE (MAPPO) · swarm comm-coverage · eval {eval_cov*100:.0f}% cover, "
           f"{eval_conn*100:.0f}% connected, max-dist {eval_maxd:.1f}"),
    fps=8, comm_radius=cc.COMM_R, metrics=metrics)
print(f"wrote {out}")
