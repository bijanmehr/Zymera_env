"""
Learning-stack template — how every piece fits together.

This is the LEARNING counterpart to the heuristic sweeper (lawn_mower.py).
It does two things:

  1. Prints every interface so you can see the actual shapes — the
     observation, the env-step 5-tuple, the info dict, and the trajectory
     pytree that `rollout` returns.
  2. Wires a complete, runnable training loop:

        env  ──reset──►  obs ─┐
                              │   ┌──────────── your learning stack ───────────┐
                              └──►│ policy → action → step → trajectory         │
                                  │ reward(traj) → loss → grad → optax update   │
                                  └─────────────────────────────────────────────┘

The algorithm here is vanilla REINFORCE with a shared multi-agent policy.
The objective is deliberately simple and LEARNABLE from the default obs
(`[row, col, coverage]`): drive the agents to the far corner. Swap the
`Policy` class, the `reward_fn`, or the `loss_fn` to drop in different
machinery; everything around them stays the same.

Note on observation: the built-in obs is just `[row, col, coverage]`. That's
enough to learn "go to the corner", but NOT enough to learn good coverage —
a coverage policy needs to know where it *hasn't* been, i.e. a richer obs
(return one from a custom env's `sense`). The obs you expose decides what is
learnable at all.

Run it:   python examples/learn_template.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

import zymera

# ── config ──────────────────────────────────────────────────────────────────
GRID, N_AGENTS, N_STEPS = 12, 2, 64      # world size · agents · episode length
ITERS, LR, GAMMA = 200, 3e-3, 0.99       # train iters · learning rate · discount
GOAL = (GRID - 1, GRID - 1)              # objective: reach the bottom-right corner


# IMPORTANT: learning runs through `rollout`, which traces the env inside
# `lax.scan` — so use the PURE-JAX env (sense_radius=0, the default). The
# partial-obs Sensor is host-side and cannot be traced; it's for the Python-loop
# agents, not for rollout.
def make_env():
    return zymera.make("empty-v0", grid_h=GRID, grid_w=GRID, n_agents=N_AGENTS)


# ============================================================================
# 1. THE INTERFACES — what the env hands you (run once, just to look)
# ============================================================================


def show_interfaces(env, key):
    obs, state = env.reset(key)
    print("reset(key) -> (obs, state)")
    print(f"  obs                 {obs.shape}  float32   # one row per agent: [row, col, coverage]")
    print(f"  state               World pytree           # the full world; leaves are JAX arrays")
    print(f"    .body.position    {tuple(state.body.position.shape)}  int32   # (row, col) per agent")
    print(f"    .explored         {tuple(state.explored.shape)}  int32   # per-cell visit count")
    print(f"    .wall             {tuple(state.wall.shape)}  bool    # obstacles")
    print(f"    .seen_by          {tuple(state.seen_by.shape)}  bool # per-agent visibility ledger")

    # one step needs an action: shape (N,) int32, one ActionId per agent
    action = jnp.full((env.n_agents,), int(zymera.ActionId.EAST), jnp.int32)
    obs, state, reward, done, info = env.step(state, action, key)
    print("\nstep(state, action, key) -> (obs, state, reward, done, info)   # the 5-tuple")
    print(f"  obs      {obs.shape}  float32   # next observation")
    print(f"  state    World pytree           # next world")
    print(f"  reward   {reward.shape}  float32   # per agent — ZEROS: reward is YOUR code, not the env's")
    print(f"  done     {done.shape}  bool      # per agent")
    print(f"  info     dict  keys={sorted(info)}")

    print(f"\nenv.n_agents={env.n_agents}  env.obs_dim={env.obs_dim}  env.n_actions={env.n_actions}"
          f"   # ActionId: STAY=0 NORTH=1 EAST=2 SOUTH=3 WEST=4")


# ============================================================================
# 2. THE POLICY — a callable obs -> action. rollout calls __call__.
# ============================================================================


class Policy(eqx.Module):
    """Shared per-agent MLP. `__call__` samples actions (what rollout uses);
    `net` also gives logits for the loss."""
    net: eqx.nn.MLP

    def __init__(self, obs_dim, n_actions, key):
        self.net = eqx.nn.MLP(obs_dim, n_actions, width_size=64, depth=2, key=key)

    def __call__(self, obs, key):                     # obs (N, obs_dim) -> (N,) int32
        logits = jax.vmap(self.net)(obs)              # (N, n_actions)
        return jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)


# ============================================================================
# 3. THE REWARD — your objective, computed from the trajectory (NOT the env)
# ============================================================================


def reward_fn(traj):
    """Per-agent reward: how much closer to the GOAL corner each step got.
    (T, N) float32. Swap this whole function for your real objective."""
    pos = traj["world"].body.position.astype(jnp.float32)        # (T+1, N, 2)
    dist = jnp.abs(pos - jnp.array(GOAL, jnp.float32)).sum(-1)   # (T+1, N) Manhattan
    return dist[:-1] - dist[1:]                                  # (T, N) closer = positive


# ============================================================================
# 4. THE LOSS — REINFORCE: pull up actions that led to high return
# ============================================================================


def _discounted_returns(rewards, gamma):
    """G[t] = sum_{k>=t} gamma^(k-t) * rewards[k].  (T, N) -> (T, N)"""
    def step(carry, r):
        carry = r + gamma * carry
        return carry, carry
    _, g = jax.lax.scan(step, jnp.zeros_like(rewards[0]), rewards[::-1])
    return g[::-1]


def loss_fn(policy, traj):
    obs = traj["obs"][:-1]                            # (T, N, obs_dim) — obs each action saw
    act = traj["action"]                              # (T, N)          — action taken
    T, N, od = obs.shape

    logits = jax.vmap(policy.net)(obs.reshape(T * N, od)).reshape(T, N, -1)   # (T, N, A)
    logp = jax.nn.log_softmax(logits, axis=-1)
    chosen = jnp.take_along_axis(logp, act[..., None], axis=-1)[..., 0]       # (T, N)

    returns = _discounted_returns(reward_fn(traj), GAMMA)                     # (T, N)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)            # baseline
    return -jnp.mean(chosen * returns)               # REINFORCE objective


# ============================================================================
# 5. HOW IT ALL COMES TOGETHER — one training step, then the loop
# ============================================================================


def main():
    key = jax.random.PRNGKey(0)
    env = make_env()

    print("=" * 78)
    show_interfaces(env, key)
    print("=" * 78)

    policy = Policy(env.obs_dim, env.n_actions, key)
    opt = optax.adam(LR)
    opt_state = opt.init(eqx.filter(policy, eqx.is_array))

    # look at the trajectory the loop consumes
    traj = zymera.rollout(env, policy, N_STEPS, key)
    print("rollout(env, policy, n_steps, key) -> traj (a dict of stacked arrays):")
    for k, v in traj.items():
        shape = v.body.position.shape if k == "world" else v.shape
        print(f"  traj[{k!r:8}]  {tuple(shape)}")
    print("=" * 78)

    @eqx.filter_jit
    def train_step(policy, opt_state, key):
        traj = zymera.rollout(env, policy, N_STEPS, key)          # collect data
        loss, grads = eqx.filter_value_and_grad(loss_fn)(policy, traj)
        updates, opt_state = opt.update(grads, opt_state)
        policy = eqx.apply_updates(policy, updates)
        final = traj["world"].body.position[-1].astype(jnp.float32)   # (N, 2)
        avg_goal_dist = jnp.abs(final - jnp.array(GOAL, jnp.float32)).sum(-1).mean()
        return policy, opt_state, loss, avg_goal_dist

    print(f"training: {ITERS} iters · grid {GRID}x{GRID} · {N_AGENTS} agents · {N_STEPS} steps/episode")
    print(f"objective: reach corner {GOAL} — avg distance should fall toward 0\n")
    for it in range(ITERS):
        key, sk = jax.random.split(key)
        policy, opt_state, loss, avg_goal_dist = train_step(policy, opt_state, sk)
        if it % 20 == 0 or it == ITERS - 1:
            print(f"  iter {it:3d}   loss {float(loss):+.4f}   "
                  f"avg dist to goal {float(avg_goal_dist):5.2f}   (0 = arrived)")


if __name__ == "__main__":
    main()
