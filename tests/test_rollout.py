"""``zymera.rollout(env, policy, n_steps, key)`` — shapes, JIT, vmap-over-keys."""

import jax
import jax.numpy as jnp

import zymera


def test_rollout_shapes():
    env = zymera.make("empty-v0", grid_h=5, grid_w=5, n_agents=2)
    n_steps = 4
    traj = zymera.rollout(env, zymera.random_policy, n_steps, jax.random.PRNGKey(0))

    N, H, W = env.n_agents, env.grid_h, env.grid_w
    # state-side: T+1 axis. action-side: T axis.
    assert traj["world"].body.position.shape == (n_steps + 1, N, 2)
    assert traj["world"].explored.shape       == (n_steps + 1, H, W)
    assert traj["obs"].shape                   == (n_steps + 1, N, env.obs_dim)
    assert traj["action"].shape                == (n_steps, N)
    assert traj["reward"].shape                == (n_steps, N)
    assert traj["done"].shape                  == (n_steps, N)


def test_rollout_initial_state_at_index_zero():
    env = zymera.make("empty-v0", grid_h=4, grid_w=4, n_agents=2)
    key = jax.random.PRNGKey(0)
    # Rollout's initial state must equal what env.reset gives for the same key.
    obs0, state0 = env.reset(jax.random.split(key)[0])
    traj = zymera.rollout(env, zymera.random_policy, 3, key)
    assert jnp.array_equal(traj["world"].explored[0], state0.explored)


def test_rollout_jit():
    env = zymera.make("empty-v0", grid_h=4, grid_w=4, n_agents=2)

    @jax.jit
    def roll(key):
        return zymera.rollout(env, zymera.random_policy, 4, key)

    traj = roll(jax.random.PRNGKey(0))
    assert traj["action"].shape == (4, env.n_agents)


def test_rollout_vmap_over_seeds():
    env = zymera.make("empty-v0", grid_h=4, grid_w=4, n_agents=2)

    def roll(k):
        return zymera.rollout(env, zymera.random_policy, 4, k)

    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    traj = jax.vmap(roll)(keys)
    assert traj["action"].shape == (3, 4, env.n_agents)
    assert traj["world"].explored.shape == (3, 5, env.grid_h, env.grid_w)
