"""
Partial-observability sensor model — the "eyes" of an agent.

A :class:`Sensor` turns a :class:`~zymera.env.World` into what a *single
agent standing in it can actually perceive*: the occupancy of nearby cells,
limited by a sense radius and (optionally) blocked by walls in the line of
sight. It is the realization of zymera's partial-observability pillar and
the producer of the world's ``seen_by`` field.

Two knobs:

* ``radius``    — how far the agent senses. A Euclidean disk by default
                  (``metric="euclidean"``): a cell is in range iff its
                  centre is within ``radius`` of the agent. ``radius=1`` is
                  the minimal 4-neighbour cross; large ``radius`` approaches
                  full observability. ``metric="chebyshev"`` makes the
                  footprint a ``(2r+1)×(2r+1)`` square instead.
* ``occlusion`` — if ``True``, walls cast shadows: a cell is only sensed
                  when the straight line from the agent to it is not blocked
                  by a wall. You see the near face of a wall but not what is
                  behind it; shadowed cells are reported as *unknown*
                  (simply absent from the reading). If ``False`` the sensor
                  is an x-ray disk — every in-range cell is sensed regardless
                  of walls.

Design note — why this is numpy, not JAX. Line-of-sight raycasting is
data-dependent control flow that does not vectorize cleanly under
``jax.lax.scan``. The sensor therefore lives at the host/numpy level and is
used in the *Python-loop* analysis path (e.g. coverage experiments), not
inside the jitted training rollout. The pure-physics ``World.step`` stays
JAX-only; the observation model is a separate, opt-in layer. See
:meth:`zymera.env.Env.step`.

::

    from zymera import Sensor, make
    env = make("empty-v0", n_obstacles=8, sense_radius=3, occlusion=True)
    obs, state = env.reset(key)
    reading = env.sensor.observe_one(state, 0)   # {(r,c): blocked_bool}
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

Cell = Tuple[int, int]


class Sensor:
    """Local occupancy sensor with adjustable range + optional occlusion.

    Stateless given a ``World``: every method is a pure function of the
    world's wall map and agent positions. The agent that consumes a reading
    only ever learns the occupancy of cells the sensor returns — never the
    grid size or the global wall array — so the local-observation contract is
    preserved structurally.
    """

    def __init__(self, radius: int = 1, occlusion: bool = False,
                 metric: str = "euclidean") -> None:
        if radius < 1:
            raise ValueError(f"sense radius must be >= 1, got {radius}")
        if metric not in ("euclidean", "chebyshev"):
            raise ValueError(f"metric must be 'euclidean' or 'chebyshev', got {metric!r}")
        self.radius = radius
        self.occlusion = occlusion
        self.metric = metric
        self._offsets = self._footprint(radius, metric)

    # ---- footprint --------------------------------------------------------

    @staticmethod
    def _footprint(r: int, metric: str) -> List[Cell]:
        """Relative ``(dr, dc)`` offsets in range, excluding the centre."""
        offs: List[Cell] = []
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if dr == 0 and dc == 0:
                    continue
                if metric == "euclidean" and dr * dr + dc * dc > r * r:
                    continue
                offs.append((dr, dc))
        return offs

    # ---- per-agent reading ------------------------------------------------

    def observe_one(self, world, i: int = 0) -> Dict[Cell, bool]:
        """What agent ``i`` perceives: ``{absolute_cell: blocked}``.

        ``blocked = wall OR off-grid``. Cells out of range, or shadowed when
        ``occlusion`` is on, are omitted (the agent treats them as unknown).
        The agent's own cell is not included (it knows it stands on free
        ground).
        """
        pos = tuple(int(x) for x in np.asarray(world.body.position[i]))
        wall = np.asarray(world.wall)
        return self._reading(pos, wall)

    def observe(self, world) -> List[Dict[Cell, bool]]:
        """:meth:`observe_one` for every agent, as a list."""
        n = world.body.position.shape[0]
        return [self.observe_one(world, i) for i in range(n)]

    def _reading(self, pos: Cell, wall: np.ndarray) -> Dict[Cell, bool]:
        H, W = wall.shape
        ar, ac = pos
        out: Dict[Cell, bool] = {}
        for dr, dc in self._offsets:
            tr, tc = ar + dr, ac + dc
            if not (0 <= tr < H and 0 <= tc < W):
                out[(tr, tc)] = True                      # off-grid reads blocked
                continue
            if self.occlusion and self._occluded(pos, (tr, tc), wall):
                continue                                  # shadowed -> unknown
            out[(tr, tc)] = bool(wall[tr, tc])
        return out

    # ---- visibility mask (feeds seen_by) ----------------------------------

    def visibility(self, world) -> np.ndarray:
        """``(N, H, W)`` bool — cells currently visible to each agent.

        Includes each agent's own cell and every sensed in-range cell
        (occlusion-aware). This is what :class:`~zymera.env.Env` accumulates
        into ``seen_by``.
        """
        wall = np.asarray(world.wall)
        H, W = wall.shape
        n = world.body.position.shape[0]
        vis = np.zeros((n, H, W), dtype=bool)
        for i in range(n):
            ar, ac = (int(x) for x in np.asarray(world.body.position[i]))
            vis[i, ar, ac] = True
            for (cr, cc) in self._reading((ar, ac), wall):
                if 0 <= cr < H and 0 <= cc < W:
                    vis[i, cr, cc] = True
        return vis

    # ---- line of sight ----------------------------------------------------

    @staticmethod
    def _occluded(a: Cell, b: Cell, wall: np.ndarray) -> bool:
        """True if a wall lies strictly between cells ``a`` and ``b``.

        Samples the segment from centre to centre densely enough to hit every
        cell it crosses; the endpoint ``b`` itself is never counted (you can
        always see the wall you are looking at, just not past it).
        """
        (ar, ac), (br, bc) = a, b
        steps = max(1, int(math.ceil(2 * math.hypot(br - ar, bc - ac))))
        for s in range(1, steps):
            t = s / steps
            cr = int(round(ar + (br - ar) * t))
            cc = int(round(ac + (bc - ac) * t))
            if (cr, cc) == a or (cr, cc) == b:
                continue
            if 0 <= cr < wall.shape[0] and 0 <= cc < wall.shape[1] and wall[cr, cc]:
                return True
        return False

    # ---- repr -------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"Sensor(radius={self.radius}, occlusion={self.occlusion}, "
                f"metric={self.metric!r})")
