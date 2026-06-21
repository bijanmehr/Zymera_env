"""Global test setup + shared fixtures."""

# Force matplotlib's non-interactive backend BEFORE any test imports
# pyplot. Tests for the live viewer and render helpers must never try
# to open a real window in CI.
import matplotlib
matplotlib.use("Agg")

import jax
import pytest

import zymera


@pytest.fixture
def key() -> jax.Array:
    return jax.random.PRNGKey(0)


@pytest.fixture
def small_world(key: jax.Array) -> zymera.World:
    """4×4 grid with 2 agents — minimum interesting world."""
    return zymera.World.initial(key, grid_h=4, grid_w=4, n_agents=2)
