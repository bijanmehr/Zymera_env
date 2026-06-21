"""Env contract — World state fields + Env wrapper + registry + Helper."""

import jax
import jax.numpy as jnp
import pytest

import zymera


# ---- World state fields (the committed contract) ---------------------------


def test_world_initial_has_committed_state_fields(small_world):
    """Every field on the World schema must be present after .initial()."""
    w = small_world
    # wired:
    assert w.body.position.shape == (w.n_agents, 2)
    assert w.body.position.dtype == jnp.int32
    assert w.explored.shape == (w.grid_h, w.grid_w)
    assert w.explored.dtype == jnp.int32
    assert w.step_count.shape == ()
    assert int(w.step_count) == 0
    # placeholder — present with correct shape, zeros today:
    assert w.seen_by.shape == (w.n_agents, w.grid_h, w.grid_w)
    assert not bool(w.seen_by.any())
    assert w.wall.shape == (w.grid_h, w.grid_w)
    assert not bool(w.wall.any())
    assert w.comm_graph.shape == (w.n_agents, w.n_agents)
    assert not bool(w.comm_graph.any())


def test_visited_property_derived_from_explored(small_world):
    """world.visited stays compat — derived from explored > 0."""
    assert int(small_world.visited.sum()) == small_world.n_agents
    assert int(small_world.explored.sum()) == small_world.n_agents


def test_step_increments_explored_and_step_count(small_world):
    action = jnp.array(
        [int(zymera.ActionId.EAST)] * small_world.n_agents,
        dtype=jnp.int32,
    )
    out = small_world.step(action)
    assert int(out.world.step_count) == 1
    assert int(out.world.explored.sum()) >= int(small_world.explored.sum())


def test_step_is_jit_compilable(small_world):
    action = jnp.array(
        [int(zymera.ActionId.NORTH)] * small_world.n_agents,
        dtype=jnp.int32,
    )
    out = jax.jit(zymera.World.step)(small_world, action)
    assert out.world.body.position.shape == (small_world.n_agents, 2)


# ---- Env wrapper + registry ------------------------------------------------


def test_registry_lists_empty_v0():
    assert "empty-v0" in zymera.list_envs()


def test_make_with_kwargs():
    env = zymera.make("empty-v0", grid_h=5, grid_w=5, n_agents=3)
    assert env.n_agents == 3
    assert env.grid_h == 5 and env.grid_w == 5
    assert env.obs_dim == 3 and env.n_actions == 5


def test_make_unknown_env_lists_available():
    with pytest.raises(ValueError, match="empty-v0"):
        zymera.make("not_a_real_env")


def test_register_duplicate_raises():
    with pytest.raises(ValueError, match="already registered"):
        zymera.register_env("empty-v0", lambda **k: None)


def test_env_reset_returns_obs_and_state(key):
    env = zymera.make("empty-v0", grid_h=4, grid_w=4, n_agents=2)
    obs, state = env.reset(key)
    assert obs.shape == (env.n_agents, env.obs_dim)
    assert isinstance(state, zymera.World)
    assert state.n_agents == env.n_agents


def test_env_step_returns_five_tuple_with_info_schema(key):
    env = zymera.make("empty-v0", grid_h=4, grid_w=4, n_agents=2)
    _, state = env.reset(key)
    action = jnp.array(
        [int(zymera.ActionId.EAST), int(zymera.ActionId.NORTH)],
        dtype=jnp.int32,
    )
    obs, state, reward, done, info = env.step(state, action, key)

    assert obs.shape == (env.n_agents, env.obs_dim)
    assert reward.shape == (env.n_agents,)
    assert done.shape == (env.n_agents,)

    # info schema committed up-front.
    assert set(info.keys()) == {"explored", "step_count", "seen_by", "comm_graph"}
    assert info["explored"].shape == (env.grid_h, env.grid_w)
    assert info["seen_by"].shape == (env.n_agents, env.grid_h, env.grid_w)
    assert info["comm_graph"].shape == (env.n_agents, env.n_agents)


# ---- Helper ----------------------------------------------------------------


def test_helper_comm_shape_symmetric_diag(small_world):
    h = zymera.Helper(comm_radius=2)
    adj = h.comm(small_world)
    N = small_world.n_agents
    assert adj.shape == (N, N)
    assert adj.dtype == jnp.bool_
    assert bool(jnp.all(adj == adj.T))
    assert bool(jnp.all(jnp.diag(adj)))


def test_helper_placeholders(small_world):
    h = zymera.Helper(comm_radius=2)
    assert not bool(h.terminate(small_world, 0).any())
    obs = small_world.sense()
    assert bool(jnp.all(h.augment(obs, small_world) == obs))


def test_helper_jit(small_world):
    h = zymera.Helper(comm_radius=2)
    assert jax.jit(h.comm)(small_world).shape == (small_world.n_agents,) * 2
