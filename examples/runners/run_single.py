"""STEP 1 reference points: ONE agent, 8x8 world (= exactly one agent's quarter of
the real 16x16 task), TIME FIXED at 70 steps (the constraint). Measures single-agent
coverage capability and pins the two reference lines every later experiment compares to:
  - DITHERING FLOOR  = the current 1-step-move policy (it re-treads ~half its steps),
  - CEILING          = a lawn-mower (column-snake), ~100% since 63 moves cover 64 cells < 70.
The arrow+commit policy (built next) must close the floor->ceiling gap.

No connectivity / no collision (one agent can't disconnect or collide), so this is a
clean read on "can it sweep efficiently." Budget math: 1 agent x (70+1) = 71 cell-visits
for 64 cells = 11% headroom -- the SAME tightness as 4 agents x 71 = 284 for 256 cells.

    python -u examples/run_single.py
"""
import statistics as stt
import sys
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc

cc.N_AGENTS = 1
cc.GRID = 8                        # 8x8 world (also makes the run banner honest)
import _marl_core as core          # noqa: E402
import _frontier_core as fc        # noqa: E402
from marl_frontier_comm import FrontierCommAttnAC as M  # noqa: E402

STEPS, CELLS, N = 70, 64, 1
cc.MAX_STEPS = STEPS
core.T = STEPS
fc.T = STEPS
core.ITERS = 200
ENV = dict(grid_h=8, grid_w=8, n_agents=1, comm_r=5, spawn_radius=2)
SHAPE = dict(w_shape1=1.0, w_shape3=2.0)
SEEDS = [1, 2, 3]


def cov_eval(env, build, vdir):
    m = eqx.tree_deserialise_leaves(str(vdir / "checkpoint.eqx"), build(env, jax.random.PRNGKey(0)))
    tr = jax.vmap(lambda k: fc._rollout_masked(env, m, STEPS, k))(
        jax.random.split(jax.random.PRNGKey(7), 32))
    st = tr["world"]
    cov = float((st.team_explored[:, -1].reshape(32, -1).sum(-1) / CELLS).mean())
    slots = N * (STEPS + 1)
    return cov, cov * CELLS / slots


def lawnmower_ceiling():
    """Empirical column-snake from a corner: walk the boustrophedon order, 1 step/turn."""
    env = cc.make_env(w_cov=1.0, w_conn=0.0, w_degree=0.0, **SHAPE, **ENV)
    H, W = 8, 8
    order = []
    for c in range(W):
        rows = range(H) if c % 2 == 0 else range(H - 1, -1, -1)
        for r in rows:
            order.append((r, c))
    _, st = env.reset(jax.random.PRNGKey(0))
    # force start at corner (0,0) for the best-case ceiling
    st = st.replace(pos=jnp.array([[0, 0]], jnp.int32),
                    explored_by=jnp.zeros_like(st.explored_by).at[0, 0, 0].set(True),
                    shared=jnp.zeros_like(st.shared).at[0, 0, 0].set(True),
                    team_explored=jnp.zeros_like(st.team_explored).at[0, 0].set(True))
    idx = 0
    _D = {(-1, 0): 1, (0, 1): 2, (1, 0): 3, (0, -1): 4}  # env _DELTAS = [STAY,N,E,S,W]
    for _ in range(STEPS):
        r, c = int(st.pos[0, 0]), int(st.pos[0, 1])
        while idx < len(order) and (r, c) == order[idx]:
            idx += 1
        if idx >= len(order):
            break
        tr_, tc = order[idx]
        dr, dc = tr_ - r, tc - c
        step = (int(jnp.sign(dr)), 0) if dr != 0 else (0, int(jnp.sign(dc)))
        a = jnp.array([_D.get(step, 0)], jnp.int32)
        _, st, _, _, _ = env.step(st, a, jax.random.PRNGKey(0))
    return float(st.team_explored.sum() / CELLS)


base = core.new_run_dir()
print(f"SINGLE-AGENT step1 · 8x8 / 1 agent / {STEPS} steps (one quarter of the real task) · {base}\n", flush=True)

VARIANTS = [
    ("baseline-1step", dict(), dict()),
    ("MEM+REVISIT", dict(mem_channels=True, w_revisit=0.3), dict()),
]
for name, ekw, mkw in VARIANTS:
    build = partial(M, **mkw)
    covs, freshes = [], []
    for sd in SEEDS:
        env = cc.make_env(w_cov=1.0, w_conn=0.0, w_degree=0.0, **SHAPE, **ekw, **ENV)
        b = fc.run(env, build, f"{name}-s{sd}", outdir=base, seed=sd,
                   target_cov=None, patience=999, min_iters=999, mask_collisions=False)
        cov, fr = cov_eval(env, build, Path(b["dir"]))
        covs.append(cov)
        freshes.append(fr)
    print(f"  ===> {name:16s} cov {stt.mean(covs)*100:5.1f}% ±{stt.pstdev(covs)*100:3.1f}  "
          f"fresh {stt.mean(freshes)*100:4.0f}%  seeds={[round(c*100, 1) for c in covs]}", flush=True)

ceil = lawnmower_ceiling()
print(f"\n  CEILING (lawn-mower column-snake from corner): {ceil*100:.1f}%", flush=True)
print(f"artifacts: {base}", flush=True)
