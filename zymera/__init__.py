"""
zymera — JAX multi-agent env library.

Use it like ``gymnax`` or ``brax``: ``make`` an env, ``reset`` / ``step``
it, write your own training loop. The core is **headless** — no plotting
libraries are imported by ``import zymera``; visualization lives in the
separate :mod:`zymera.viz` subpackage.

Quick start
-----------
::

    import jax, zymera
    from zymera import viz

    env  = zymera.make("empty-v0", grid_h=8, grid_w=8, n_agents=4)
    traj = zymera.rollout(env, zymera.random_policy, 32, jax.random.PRNGKey(0))
    viz.render_gif(traj, "rollout.gif")        # or viz.iso_gif / viz.make_report

Public API (headless core)
--------------------------
- :func:`make` / :func:`list_envs` / :func:`register_env`
- :class:`Env`           — gym-style base class (subclass for custom envs)
- :class:`GridEnv`       — the one built-in env (registered as "empty-v0")
- :func:`rollout`        — ``lax.scan``-based, JIT + vmap friendly
- :func:`random_policy`  — example uniform-action policy
- :class:`Sensor`        — partial-observability reading + ``seen_by``
- :class:`Helper`        — comm-graph utility (terminate/augment are placeholders)

Visualization (``import zymera.viz``): ``render_gif``, ``iso_gif``,
``make_report``, ``teleop``, and the ``draw_*`` primitives.
"""

from .env import (
    ActionId,
    Body,
    Env,
    GridEnv,
    N_ACTIONS,
    StepOutput,
    World,
    list_envs,
    make,
    register_env,
)
from .helper import Helper
from .sensor import Sensor
from .policies import random_policy
from .rollout import rollout

# Visualization lives in the ``zymera.viz`` subpackage — kept out of the
# headless core so ``import zymera`` pulls in no plotting libraries. Use it
# explicitly: ``from zymera import viz`` / ``import zymera.viz``.

__all__ = [
    # registry / construction
    "make", "list_envs", "register_env",
    # env classes
    "Env", "GridEnv",
    # state types (exposed so users can write type hints / custom envs)
    "World", "Body", "ActionId", "N_ACTIONS", "StepOutput",
    # helper + partial-observability sensor
    "Helper", "Sensor",
    # rollout + policies
    "rollout", "random_policy",
]
