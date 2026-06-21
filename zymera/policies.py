"""
One example policy. Users are expected to write their own.

The policy contract is: ``policy(obs, key) -> action``, where
``obs`` is ``(N, obs_dim)`` and the return is ``(N,) int32``.
"""

import jax
import jax.numpy as jnp

from .env import N_ACTIONS


def random_policy(obs: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """Sample one action per agent uniformly at random.

    ``obs`` is ignored (random_policy is stateless and observation-blind).
    ``key`` drives the sampling; caller is responsible for splitting fresh
    keys per step.

    Returns ``(N,)`` int32 with each entry in ``[0, N_ACTIONS)``.
    """
    n = obs.shape[0]
    return jax.random.randint(key, (n,), 0, N_ACTIONS, dtype=jnp.int32)
