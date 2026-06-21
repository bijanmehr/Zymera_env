"""Scripted potential-field arrow — NO learning. Validates the arrow math and gives
the pure-heuristic reference: a single agent that, each step, computes a "toward the
uncovered mass" vector and steps along it (greedy). Expected to land near the ~70%
greedy cap — the line the LEARNED arrow+correction must beat.

Single agent, 8x8, 70 steps (= one quarter of the real task).

    python -u examples/arrow_probe.py
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
import comm_coverage as cc

cc.N_AGENTS = 1
cc.GRID = 8
STEPS, CELLS = 70, 64
ENV = dict(grid_h=8, grid_w=8, n_agents=1, comm_r=5, spawn_radius=2)
# env _DELTAS = [STAY, N(-1,0), E(0,1), S(1,0), W(0,-1)] -> dir actions 1..4
_DIRS = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], jnp.float32)


def agent_arrow(p, known, wall, H, W):
    """Potential-field pull toward uncovered cells (inverse-square weighted). (2,)."""
    ys = jnp.arange(H, dtype=jnp.float32)[:, None] * jnp.ones((1, W))
    xs = jnp.arange(W, dtype=jnp.float32)[None, :] * jnp.ones((H, 1))
    dy = ys - p[0].astype(jnp.float32)
    dx = xs - p[1].astype(jnp.float32)
    dist2 = dy * dy + dx * dx + 1e-3
    dist = jnp.sqrt(dist2)
    uncovered = ((~known) & (~wall)).astype(jnp.float32)
    w = uncovered / dist2                              # nearer uncovered pulls harder
    ay = (w * dy / dist).sum()
    ax = (w * dx / dist).sum()
    return jnp.array([ay, ax])


def arrow_to_action(arrow):
    norm = jnp.sqrt((arrow * arrow).sum())
    best = jnp.argmax(_DIRS @ arrow) + 1               # 1..4 = N,E,S,W
    return jnp.where(norm < 1e-4, 0, best).astype(jnp.int32)


def scripted_rollout(env, T, key):
    rk, sk = jax.random.split(key)
    _, st0 = env.reset(rk)

    def body(st, k):
        arrow = agent_arrow(st.pos[0], st.shared[0], st.wall, env.H, env.W)
        a = arrow_to_action(arrow)[None]               # (1,)
        _, st2, _, _, _ = env.step(st, a, k)
        return st2, st2.team_explored.sum()

    keys = jax.random.split(sk, T)
    _, covered = jax.lax.scan(body, st0, keys)
    return covered[-1] / CELLS                          # final coverage fraction


env = cc.make_env(w_cov=1.0, w_conn=0.0, w_degree=0.0, w_shape1=1.0, w_shape3=2.0, **ENV)
keys = jax.random.split(jax.random.PRNGKey(7), 64)
covs = jax.vmap(lambda k: scripted_rollout(env, STEPS, k))(keys)
print(f"SCRIPTED potential-field (greedy, no learning) · 1 agent / 8x8 / {STEPS} steps")
print(f"  coverage: {float(covs.mean())*100:.1f}%  (min {float(covs.min())*100:.0f}% / "
      f"max {float(covs.max())*100:.0f}%, over 64 episodes)")
print(f"  reference lines: dithering 1-step learned ~?? · this heuristic · lawn-mower ~100%")
