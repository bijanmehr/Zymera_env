"""Tests for the partial-observability layer: obstacles, wall enforcement,
the Sensor (radius + occlusion), and seen_by wiring."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import zymera
from zymera import Sensor


# ---- obstacles + wall enforcement -----------------------------------------


def test_movement_blocked_by_wall():
    """An agent cannot step onto a wall cell; it stays put."""
    wall = np.zeros((5, 5), bool)
    wall[0, 1] = True                                   # east of (0,0)
    env = zymera.make("empty-v0", grid_h=5, grid_w=5, n_agents=1, wall=wall)
    # find a key that spawns at (0,0)
    for s in range(100):
        _, state = env.reset(jax.random.PRNGKey(s))
        if tuple(int(x) for x in state.body.position[0]) == (0, 0):
            break
    _, state2, _, _, _ = env.step(
        state, jnp.array([int(zymera.ActionId.EAST)]), jax.random.PRNGKey(0))
    assert tuple(int(x) for x in state2.body.position[0]) == (0, 0)


def test_spawn_avoids_walls():
    """Agents never spawn on an obstacle."""
    wall = np.zeros((6, 6), bool)
    wall[:, 0] = True                                   # whole first column walled
    env = zymera.make("empty-v0", grid_h=6, grid_w=6, n_agents=3, wall=wall)
    for s in range(20):
        _, state = env.reset(jax.random.PRNGKey(s))
        for i in range(3):
            r, c = (int(x) for x in state.body.position[i])
            assert not bool(wall[r, c])


def test_open_grid_movement_unchanged():
    """With no walls the wall check is a no-op (regression guard)."""
    env = zymera.make("empty-v0", grid_h=5, grid_w=5, n_agents=1)
    for s in range(10):
        _, state = env.reset(jax.random.PRNGKey(s))
        r, c = (int(x) for x in state.body.position[0])
        if c < 4:
            break
    _, state2, _, _, _ = env.step(
        state, jnp.array([int(zymera.ActionId.EAST)]), jax.random.PRNGKey(0))
    r2, c2 = (int(x) for x in state2.body.position[0])
    assert (r2, c2) == (r, c + 1)


# ---- sensor footprint ------------------------------------------------------


@pytest.mark.parametrize("radius,expected", [(1, 4), (2, 12), (3, 28)])
def test_euclidean_footprint_size(radius, expected):
    assert len(Sensor(radius=radius)._offsets) == expected


def test_chebyshev_footprint_is_square():
    # (2r+1)^2 - 1 cells
    assert len(Sensor(radius=2, metric="chebyshev")._offsets) == 24


def test_radius_must_be_positive():
    with pytest.raises(ValueError):
        Sensor(radius=0)


# ---- sensor readings -------------------------------------------------------


def _world_at(pos, wall):
    """Minimal duck-typed world for sensor unit tests."""
    class _W:
        class body:
            position = np.array([list(pos)])
    _W.wall = np.asarray(wall)
    return _W


def test_off_grid_reads_blocked():
    wall = np.zeros((3, 3), bool)
    reading = Sensor(radius=1).observe_one(_world_at((0, 0), wall), 0)
    assert reading[(-1, 0)] is True                     # north off-grid
    assert reading[(0, -1)] is True                     # west off-grid
    assert reading[(1, 0)] is False                     # south, free


def test_occlusion_shadows_cells_behind_walls():
    wall = np.zeros((1, 5), bool)
    wall[0, 2] = True
    xray = Sensor(radius=4, occlusion=False).observe_one(_world_at((0, 0), wall), 0)
    occ = Sensor(radius=4, occlusion=True).observe_one(_world_at((0, 0), wall), 0)
    assert (0, 3) in xray                               # x-ray sees behind the wall
    assert (0, 3) not in occ                            # occlusion hides it
    assert occ[(0, 2)] is True                          # but the wall face is visible


def test_occlusion_diagonal_block():
    wall = np.zeros((3, 3), bool)
    wall[1, 1] = True
    occ = Sensor(radius=2, occlusion=True).observe_one(_world_at((0, 0), wall), 0)
    assert (2, 2) not in occ                            # (1,1) shadows (2,2)


# ---- seen_by wiring --------------------------------------------------------


def test_seen_by_wired_and_accumulates():
    env = zymera.make("empty-v0", grid_h=8, grid_w=8, n_agents=1,
                      sense_radius=2, occlusion=False)
    _, state = env.reset(jax.random.PRNGKey(0))
    seen0 = int(np.asarray(state.seen_by).sum())
    assert seen0 > 1                                    # sees a disk at spawn
    _, state, _, _, info = env.step(
        state, jnp.array([int(zymera.ActionId.SOUTH)]), jax.random.PRNGKey(0))
    seen1 = int(np.asarray(state.seen_by).sum())
    assert seen1 >= seen0                               # monotone accumulation


def test_no_sensor_keeps_seen_by_placeholder():
    """Default env (sense_radius=0) leaves seen_by zeros and stays jittable."""
    env = zymera.make("empty-v0", grid_h=6, grid_w=6, n_agents=2)
    assert env.sensor is None
    _, state = env.reset(jax.random.PRNGKey(0))
    assert int(np.asarray(state.seen_by).sum()) == 0
    # rollout drives env.step inside lax.scan — must still trace
    traj = zymera.rollout(env, zymera.random_policy, 8, jax.random.PRNGKey(0))
    assert traj["world"].body.position.shape[0] == 9
