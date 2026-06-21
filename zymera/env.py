"""
Env + state + registry — the env half of the library.

Public surface (re-exported by :mod:`zymera`):
    :class:`Env`         — gym-style base class (reset / step).
    :class:`GridEnv`     — the one built-in env (registered as "empty-v0").
    :func:`make`         — registry factory: ``make(name, **kwargs) → Env``.
    :func:`list_envs`    — registered env names.
    :func:`register_env` — add a custom env.

Internals (callers usually don't need to import these directly, but
they're public for advanced use):
    :class:`World`       — the env state pytree (one per running env).
    :class:`Body`        — per-agent physical state.
    :class:`ActionId`    — movement enum.
    ``N_ACTIONS``        — len(ActionId).
"""

from enum import IntEnum
from typing import Any, Callable, Dict, Tuple

import chex
import jax
import jax.numpy as jnp


# =============================================================================
# Actions
# =============================================================================


class ActionId(IntEnum):
    """Movement vocabulary on the square grid.

    The action emitted by a policy for agent ``i`` is an int in
    ``[0, N_ACTIONS)``. Integer values are stable — extend by appending
    so existing checkpoints keep their meaning.
    """

    STAY = 0
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4


N_ACTIONS = len(ActionId)


# Per-action ``(dr, dc)`` delta. Dict keyed by enum so each entry is
# self-documenting; the assert below the dict catches drift at import time.
_DELTA_BY_ACTION = {
    ActionId.STAY:  (0,  0),
    ActionId.NORTH: (-1, 0),
    ActionId.EAST:  (0,  1),
    ActionId.SOUTH: (1,  0),
    ActionId.WEST:  (0, -1),
}
assert set(_DELTA_BY_ACTION) == set(ActionId), (
    "every ActionId needs an entry in _DELTA_BY_ACTION (and vice versa)"
)

# Materialised lookup table indexed by action int — iteration over IntEnum is
# value-ordered, so ``ACTION_DELTAS[a]`` is always agent ``a``'s delta.
ACTION_DELTAS = jnp.array(
    [_DELTA_BY_ACTION[a] for a in ActionId],
    dtype=jnp.int32,
)


# =============================================================================
# State pytrees
# =============================================================================


@chex.dataclass(frozen=True)
class Body:
    """Per-agent physical state. SoA — every field is shape ``(N, ...)``."""

    position: chex.Array        # (N, 2) int32  — (row, col)


@chex.dataclass(frozen=True)
class StepOutput:
    """Bundle returned by ``World.step``."""

    world:  "World"
    obs:    chex.Array          # (N, obs_dim) float
    reward: chex.Array          # (N,) — zeros placeholder; reward is user code
    done:   chex.Array          # (N,) bool — zeros placeholder; user / helper code


@chex.dataclass(frozen=True)
class World:
    """The simulated env's state.

    Immutable JAX pytree — every "mutation" returns a fresh ``World``.
    Threads through ``jax.jit`` / ``jax.vmap`` / ``jax.lax.scan`` cleanly.

    State fields are the committed contract user reward / mission code
    reads from. Some fields are *wired* (compute real values); some are
    *placeholder* (zeros) until the surrounding logic ships. The schema
    is stable either way — user code written today keeps working when
    placeholders fill in.
    """

    # ---- wired today -----------------------------------------------------
    body:        Body           # (N, 2) — agent positions
    explored:    chex.Array     # (H, W) int32 — per-cell visit count
    step_count:  chex.Array     # ()     int32 — ticks since reset

    # ---- placeholder until surrounding logic ships -----------------------
    seen_by:     chex.Array     # (N, H, W) bool — per-agent visibility
    wall:        chex.Array     # (H, W)    bool — env physics
    comm_graph:  chex.Array     # (N, N)    bool — adjacency this step

    # ---- constructor ------------------------------------------------------

    @classmethod
    def initial(cls, key: jax.Array, grid_h: int, grid_w: int,
                n_agents: int, wall: jax.Array = None) -> "World":
        """Build a fresh World; agents placed at random *free* cells.

        ``wall`` is an optional ``(H, W)`` bool obstacle mask; when omitted
        the grid is open. Spawns avoid wall cells (sampling is weighted to
        free cells), and spawn cells get one count in ``explored`` so the
        first rollout step sees coherent state.
        """
        if wall is None:
            wall = jnp.zeros((grid_h, grid_w), dtype=jnp.bool_)
        else:
            wall = jnp.asarray(wall, dtype=jnp.bool_)

        # Spawn on distinct free cells: sample flat indices without
        # replacement, weighted to free cells (~wall).
        free = (~wall).reshape(-1).astype(jnp.float32)
        probs = free / jnp.sum(free)
        flat = jax.random.choice(key, grid_h * grid_w, shape=(n_agents,),
                                 replace=False, p=probs)
        rows = (flat // grid_w).astype(jnp.int32)
        cols = (flat %  grid_w).astype(jnp.int32)
        position = jnp.stack([rows, cols], axis=-1).astype(jnp.int32)

        body = Body(position=position)
        explored   = jnp.zeros((grid_h, grid_w), dtype=jnp.int32).at[rows, cols].add(1)
        seen_by    = jnp.zeros((n_agents, grid_h, grid_w), dtype=jnp.bool_)
        comm_graph = jnp.zeros((n_agents, n_agents), dtype=jnp.bool_)
        step_count = jnp.zeros((), dtype=jnp.int32)

        return cls(
            body=body, explored=explored, step_count=step_count,
            seen_by=seen_by, wall=wall, comm_graph=comm_graph,
        )

    # ---- shape helpers ----------------------------------------------------

    @property
    def grid_h(self) -> int:
        return self.explored.shape[0]

    @property
    def grid_w(self) -> int:
        return self.explored.shape[1]

    @property
    def n_agents(self) -> int:
        return self.body.position.shape[0]

    @property
    def visited(self) -> chex.Array:
        """``True`` where any agent has stepped — derived from ``explored > 0``."""
        return self.explored > 0

    # ---- read-only sensor -------------------------------------------------

    def sense(self) -> chex.Array:
        """Per-agent obs for the *current* state, no step taken."""
        return self._sensing(self.body, self.explored)

    # ---- step subsystems (private) ----------------------------------------

    def _movement(self, action: chex.Array) -> Body:
        """Apply per-agent actions; clip at the boundary and block walls.

        A move into the boundary is clipped to in-bounds; a move into a
        wall cell is rejected (the agent stays put). Both are pure JAX, so
        this stays ``lax.scan`` / ``jit`` / ``vmap`` safe. With an open grid
        (``wall`` all-False) the wall check is a no-op.
        """
        proposed = self.body.position + ACTION_DELTAS[action]
        h, w = self.explored.shape
        row = jnp.clip(proposed[..., 0], 0, h - 1)
        col = jnp.clip(proposed[..., 1], 0, w - 1)
        hits_wall = self.wall[row, col]                          # (N,) bool
        cur_row, cur_col = self.body.position[..., 0], self.body.position[..., 1]
        row = jnp.where(hits_wall, cur_row, row)
        col = jnp.where(hits_wall, cur_col, col)
        return self.body.replace(position=jnp.stack([row, col], axis=-1))

    def _terrain(self, new_body: Body) -> chex.Array:
        """Increment per-cell visit counts for the new positions."""
        row, col = new_body.position[:, 0], new_body.position[:, 1]
        return self.explored.at[row, col].add(1)

    def _sensing(self, new_body: Body, new_explored: chex.Array) -> chex.Array:
        """Per-agent obs: own ``(row, col)`` + global coverage fraction."""
        n = new_body.position.shape[0]
        pos = new_body.position.astype(jnp.float32)
        coverage = jnp.mean((new_explored > 0).astype(jnp.float32))
        coverage_col = jnp.broadcast_to(coverage, (n, 1))
        return jnp.concatenate([pos, coverage_col], axis=-1)

    # ---- public step ------------------------------------------------------

    def step(self, action: chex.Array) -> StepOutput:
        """Run one env tick. Returns a fresh ``World`` inside ``StepOutput``."""
        new_body = self._movement(action)
        new_explored = self._terrain(new_body)
        obs = self._sensing(new_body, new_explored)

        new_world = self.replace(
            body=new_body, explored=new_explored,
            step_count=self.step_count + 1,
        )

        n = new_body.position.shape[0]
        reward = jnp.zeros((n,), dtype=jnp.float32)
        done = jnp.zeros((n,), dtype=jnp.bool_)
        return StepOutput(world=new_world, obs=obs, reward=reward, done=done)


# =============================================================================
# Gym-style Env wrapper
# =============================================================================


class Env:
    """Gym-style env wrapper around ``World``.

    Subclasses configure the underlying World (grid size, n_agents,
    future scenario parameters) in ``__init__`` + ``reset``. ``step`` is
    inherited and works for any subclass whose ``state`` is a ``World``.

    Class attributes ``obs_dim`` and ``n_actions`` are the env's
    contract for policy authors.
    """

    obs_dim:   int = 3
    n_actions: int = N_ACTIONS

    def __init__(self, n_agents: int, sensor: "Any" = None):
        self.n_agents = n_agents
        # Optional partial-observability model. When present, ``step``/``reset``
        # run it (host-side numpy) to fill ``seen_by`` and expose the current
        # visibility in ``info``. None keeps the env pure-JAX so ``rollout``
        # (which calls ``step`` inside ``lax.scan``) stays jittable/vmappable.
        self.sensor = sensor

    # --- gym-style API -----------------------------------------------------

    def reset(self, key: jax.Array) -> Tuple[jax.Array, "World"]:
        """Build a fresh world; return ``(obs, state)``."""
        raise NotImplementedError("subclass must implement reset()")

    def step(
        self, state: "World", action: jax.Array, key: jax.Array,
    ) -> Tuple[jax.Array, "World", jax.Array, jax.Array, Dict[str, Any]]:
        """Apply ``action``; return ``(obs, state', reward, done, info)``."""
        del key   # current envs are deterministic; key reserved for stochastic envs
        out = state.step(action)
        world = self._observe_into(out.world)
        info = _info_dict(world)
        return out.obs, world, out.reward, out.done, info

    # --- partial-observability hook ----------------------------------------

    def _observe_into(self, world: "World") -> "World":
        """Accumulate the sensor's current visibility into ``seen_by``.

        No-op (returns ``world`` unchanged) when the env has no sensor.
        """
        if self.sensor is None:
            return world
        vis = jnp.asarray(self.sensor.visibility(world))   # (N, H, W) bool
        return world.replace(seen_by=world.seen_by | vis)

    def _reset_observe(self, state: "World") -> "World":
        """Seed ``seen_by`` from what the agents see at spawn."""
        if self.sensor is None:
            return state
        vis = jnp.asarray(self.sensor.visibility(state))
        return state.replace(seen_by=vis)


def _info_dict(world: "World") -> Dict[str, Any]:
    """The ``info`` schema returned from every ``env.step``.

    Keys are committed up-front; values come from World fields. ``seen_by``
    is *wired* when the env has a :class:`~zymera.sensor.Sensor` (opt in via
    ``sense_radius``) and the zero placeholder otherwise; ``comm_graph`` is
    still a placeholder. User reward / mission code reads these names from
    day one regardless.
    """
    return {
        "explored":   world.explored,     # (H, W)    int32 — wired
        "step_count": world.step_count,   # ()        int32 — wired
        "seen_by":    world.seen_by,      # (N, H, W) bool  — wired iff sensor attached
        "comm_graph": world.comm_graph,   # (N, N)    bool  — placeholder
    }


# =============================================================================
# Built-in envs
# =============================================================================


def _random_wall(key: jax.Array, grid_h: int, grid_w: int,
                 n_obstacles: int) -> jax.Array:
    """``(H, W)`` bool mask with ``n_obstacles`` random cells set True."""
    n = min(n_obstacles, grid_h * grid_w)
    idx = jax.random.choice(key, grid_h * grid_w, shape=(n,), replace=False)
    flat = jnp.zeros((grid_h * grid_w,), dtype=jnp.bool_).at[idx].set(True)
    return flat.reshape(grid_h, grid_w)


class GridEnv(Env):
    """Square grid env, optionally with obstacles and a sensor.

    ``zymera.make("empty-v0", grid_h=H, grid_w=W, n_agents=N)`` is the open
    grid. Opt into obstacles and partial observability::

        env = zymera.make("empty-v0", grid_h=16, grid_w=16, n_agents=1,
                          n_obstacles=20, sense_radius=3, occlusion=True)

    Parameters
    ----------
    n_obstacles : int
        Number of random wall cells, regenerated from the reset key each
        episode. Ignored if ``wall`` is given.
    wall : array-like or None
        Explicit ``(H, W)`` bool obstacle mask (fixed across resets). Use
        for controlled / benchmark maps.
    sense_radius : int
        ``0`` (default) → no sensor, env stays pure-JAX (``rollout`` safe).
        ``>= 1`` → attach a :class:`~zymera.sensor.Sensor`, wiring ``seen_by``
        and ``info["visible"]`` (host-side; use the Python-loop API, not
        ``lax.scan``).
    occlusion : bool
        Whether walls block the sensor's line of sight. Only meaningful when
        ``sense_radius >= 1``.
    """

    def __init__(self, grid_h: int = 8, grid_w: int = 8, n_agents: int = 1,
                 n_obstacles: int = 0, wall=None,
                 sense_radius: int = 0, occlusion: bool = False):
        sensor = None
        if sense_radius and sense_radius >= 1:
            from .sensor import Sensor
            sensor = Sensor(radius=sense_radius, occlusion=occlusion)
        super().__init__(n_agents=n_agents, sensor=sensor)
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_obstacles = n_obstacles
        self._wall = None if wall is None else jnp.asarray(wall, dtype=jnp.bool_)

    def reset(self, key: jax.Array) -> Tuple[jax.Array, "World"]:
        wall_key, spawn_key = jax.random.split(key)
        if self._wall is not None:
            wall = self._wall
        elif self.n_obstacles > 0:
            wall = _random_wall(wall_key, self.grid_h, self.grid_w, self.n_obstacles)
        else:
            wall = None
        state = World.initial(spawn_key, self.grid_h, self.grid_w,
                              self.n_agents, wall=wall)
        state = self._reset_observe(state)
        return state.sense(), state


# =============================================================================
# Registry
# =============================================================================

_REGISTRY: Dict[str, Callable[..., Env]] = {}


def register_env(name: str, factory: Callable[..., Env]) -> None:
    """Register ``factory(**kwargs) -> Env`` under ``name``."""
    if name in _REGISTRY:
        raise ValueError(f"env name already registered: {name!r}")
    _REGISTRY[name] = factory


def list_envs() -> list[str]:
    """All registered env names, sorted."""
    return sorted(_REGISTRY.keys())


def make(name: str, **kwargs) -> Env:
    """Construct a registered env by name + optional kwargs.

    ::

        env = zymera.make("empty-v0", grid_h=8, grid_w=8, n_agents=4)
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown env: {name!r}; available: {list_envs()}"
        )
    return _REGISTRY[name](**kwargs)


# ---- default registry entries -----------------------------------------------

register_env("empty-v0", GridEnv)
