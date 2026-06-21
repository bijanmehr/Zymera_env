"""Graph-belief swarm — prototype.

A host-side (numpy) prototype of the graph-belief swarm design (see
`graph-belief-swarm-whitepaper.html` at the workspace root). Built on top of
the zymera `GridEnv` + `Sensor` for world physics and partial observability.

Three modules per agent (whitepaper framing):
  * COMPASS         — goal selection over the belief (uncertainty - crowding).
  * SLAM CONTROLLER — build the belief map + path-follow to the goal (BFS).
  * EMERGENCE       — between-agent coupling: belief-merge + claims on contact.

This v1 uses a *deterministic* compass (no learning) so the emergence mechanism
can be observed and ablated immediately. The learned compass is the next step.
"""

from .belief import Belief
from .swarm import run_episode

__all__ = ["Belief", "run_episode"]
