"""
``rollout(env, policy, n_steps, key)`` — scan-based rollout primitive.

JIT-pure, ``vmap``-friendly. Wraps the env's ``reset`` + ``step`` in
one ``lax.scan``; returns a stacked-pytree trajectory with leading
``T + 1`` axis on state-side leaves and ``T`` on action-side leaves.

For interactive / breakpoint-friendly stepping see
:func:`zymera.viz.teleop` (matplotlib window) or just call
``env.reset`` / ``env.step`` directly in a Python for-loop.
"""

from typing import Callable, Dict

import jax
import jax.numpy as jnp


def rollout(env, policy: Callable, n_steps: int, key: jax.Array) -> Dict:
    """Roll ``env`` forward ``n_steps`` ticks under ``policy``.

    Parameters
    ----------
    env       : :class:`zymera.Env` instance.
    policy    : callable ``(obs, key) -> (N,) int32 action``.
    n_steps   : static int — episode length.
    key       : PRNGKey; split internally for reset + per-step subkeys.

    Returns
    -------
    Dict with keys:

      * ``"world"``  — stacked World pytree, leaves shape ``(T+1, ...)``
      * ``"obs"``    — ``(T+1, N, obs_dim)``
      * ``"action"`` — ``(T,   N)``     int32
      * ``"reward"`` — ``(T,   N)``     float32
      * ``"done"``   — ``(T,   N)``     bool

    Vmap-friendly — wrap with ``jax.vmap`` over the ``key`` arg for free
    seed-parallelism::

        keys = jax.random.split(key, 32)
        trajs = jax.vmap(lambda k: rollout(env, policy, 100, k))(keys)
    """
    reset_key, scan_key = jax.random.split(key)
    obs0, state0 = env.reset(reset_key)

    def body(carry, _):
        state, obs, k = carry
        k, action_key, step_key = jax.random.split(k, 3)
        action = policy(obs, action_key)
        obs_next, state_next, reward, done, _info = env.step(state, action, step_key)
        per_step = {
            "world":  state_next,
            "obs":    obs_next,
            "action": action,
            "reward": reward,
            "done":   done,
        }
        return (state_next, obs_next, k), per_step

    _, stacked = jax.lax.scan(
        body, (state0, obs0, scan_key), xs=None, length=n_steps,
    )

    # Prepend the initial snapshot so state-side leaves have a (T+1, ...)
    # axis and indexing matches the obvious "index 0 is pre-step state"
    # convention.
    init_state_stacked = jax.tree_util.tree_map(lambda x: x[None, ...], state0)
    full_state = jax.tree_util.tree_map(
        lambda init, rest: jnp.concatenate([init, rest], axis=0),
        init_state_stacked,
        stacked["world"],
    )
    full_obs = jnp.concatenate([obs0[None, ...], stacked["obs"]], axis=0)

    return {
        "world":  full_state,
        "obs":    full_obs,
        "action": stacked["action"],
        "reward": stacked["reward"],
        "done":   stacked["done"],
    }
