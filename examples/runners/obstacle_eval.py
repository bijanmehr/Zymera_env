"""Zero-shot generalization of a learnt policy (trained clustered-start, NO
obstacles) to maps WITH obstacles — "use the learnt policy and try variations
of maps with obstacles, see how it works."

Loads a saved checkpoint and rolls it out on random obstacle maps at increasing
densities (many seeds each, clustered start to match the training distribution).
Coverage is measured over FREE cells (the wall cells can never be covered; some
free cells may also be unreachable behind walls, so even a perfect policy won't
hit 100%). Reports coverage / connectivity / spread / collisions vs density, and
renders one representative obstacle episode as an HTML report.

    python examples/obstacle_eval.py experiments/marl/<ts> [independent|ctde|joint]
"""

import csv
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
from marl_independent import IndependentAC
from marl_ctde import CTDE_AC
from marl_joint import JointAC

MODELS = {"independent": IndependentAC, "ctde": CTDE_AC, "joint": JointAC}
DENSITIES = [0, 8, 16, 24, 32, 48]   # obstacle counts (of 256 cells → 0–19%)
SEEDS = 32                            # random maps per density
VIZ_DENSITY = 24                      # which density to render a representative episode for

run_dir = Path(sys.argv[1])
variant = sys.argv[2] if len(sys.argv) > 2 else "independent"
vdir = run_dir / variant
Model = MODELS[variant]

# ---- load the learnt policy ------------------------------------------------
like = Model(cc.make_env(), jax.random.PRNGKey(0))
model = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"), like)

def policy(obs, key):
    return jax.random.categorical(key, model.logits(obs), axis=-1).astype(jnp.int32)

off = ~jnp.eye(cc.N_AGENTS, dtype=bool)

def eval_density(k, n_seeds, base_seed):
    env = cc.make_env(n_obstacles=k)
    keys = jax.random.split(jax.random.PRNGKey(base_seed), n_seeds)
    trajs = jax.vmap(lambda kk: zymera.rollout(env, policy, cc.MAX_STEPS, kk))(keys)
    st = trajs["world"]
    wall = st.wall[:, -1].reshape(n_seeds, -1)                     # (S, H*W)
    free = jnp.maximum((~wall).sum(-1), 1)                         # free cells per map
    covered = st.team_explored[:, -1].reshape(n_seeds, -1).sum(-1)
    cover_free = covered / free
    pos = st.pos                                                  # (S, T+1, N, 2)
    d = jnp.max(jnp.abs(pos[..., :, None, :] - pos[..., None, :, :]), -1)
    conn = jax.vmap(jax.vmap(lambda dd: cc._connected(dd <= cc.COMM_R)))(d).mean(-1)  # over steps
    maxd = jnp.where(off, d, 0).max((-1, -2))[:, -1]              # final-step spread
    coll = (((d == 0) & off).sum((-1, -2)) / 2.0).mean(-1)        # mean collision-pairs/step
    return (float(cover_free.mean()), float(cover_free.std()),
            float(conn.mean()), float(maxd.mean()), float(coll.mean()))

print(f"zero-shot generalization · {variant} policy from {vdir}")
print(f"(trained clustered-start, 0 obstacles; coverage measured over FREE cells)\n")
print(f"{'obstacles':>9} {'%grid':>6} {'cover(free)':>13} {'connected':>10} "
      f"{'max-dist':>9} {'collisions':>11}")
rows = []
for k in DENSITIES:
    cf, sd, cn, md, cl = eval_density(k, SEEDS, 100 + k)
    print(f"{k:9d} {100*k/(cc.GRID**2):5.1f}% {cf*100:11.1f}%±{sd*100:3.0f} "
          f"{cn*100:9.0f}% {md:9.1f} {cl:11.2f}")
    rows.append([k, round(100*k/(cc.GRID**2), 1), round(cf, 4), round(sd, 4),
                 round(cn, 4), round(md, 4), round(cl, 4)])

with open(vdir / "obstacle_generalization.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_obstacles", "pct_grid", "cover_free", "cover_free_sd",
                "connected", "max_dist", "collisions"])
    w.writerows(rows)

# ---- render one representative obstacle episode ----------------------------
env = cc.make_env(n_obstacles=VIZ_DENSITY)
keys = jax.random.split(jax.random.PRNGKey(7), SEEDS)
trajs = jax.vmap(lambda kk: zymera.rollout(env, policy, cc.MAX_STEPS, kk))(keys)
st = trajs["world"]
wall = st.wall[:, -1].reshape(SEEDS, -1)
cover_free = st.team_explored[:, -1].reshape(SEEDS, -1).sum(-1) / jnp.maximum((~wall).sum(-1), 1)
rep = int(np.argmin(np.abs(np.asarray(cover_free) - float(cover_free.mean()))))
ep = jax.tree_util.tree_map(lambda x: x[rep], st)
n_frames = int(ep.pos.shape[0])

def snap(t):
    s = jax.tree_util.tree_map(lambda x: x[t], ep)
    return types.SimpleNamespace(
        visited=np.asarray(s.team_explored),
        explored=np.asarray(s.explored_by).sum(0),
        wall=np.asarray(s.wall),
        body=types.SimpleNamespace(position=np.asarray(s.pos)),
    )

viz_traj = {"world": [snap(t) for t in range(n_frames)]}
out = vdir / "obstacle_report.html"
viz.make_report(viz_traj, str(out),
                title=(f"{variant} policy · zero-shot on {VIZ_DENSITY} obstacles "
                       f"({100*VIZ_DENSITY/(cc.GRID**2):.0f}% of grid)"),
                fps=8, comm_radius=cc.COMM_R)
print(f"\nwrote {vdir / 'obstacle_generalization.csv'}")
print(f"wrote {out}  (representative episode, {VIZ_DENSITY} obstacles)")
