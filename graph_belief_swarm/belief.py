"""Per-agent relational belief over the grid (numpy, host-side).

The belief is the agent's persistent knowledge base. v1 keeps it as two
boolean maps over (H, W); the typed-graph schema in the whitepaper (cells /
agents / obstacles as node types) is the planned extension. Wall and free
cells are *derived* from `observed` against the known occupancy, so the agent
only "knows" a wall where it (or a teammate, via merge) has actually sensed.

The merge operation -- elementwise OR of two beliefs -- is the graph-union
"share on contact" that the Emergence layer performs between in-range agents.
"""

from __future__ import annotations

import numpy as np


class Belief:
    def __init__(self, H: int, W: int, wall_truth: np.ndarray):
        self.H, self.W = H, W
        self._wall_truth = np.asarray(wall_truth, dtype=bool)   # ground-truth occupancy
        self.observed = np.zeros((H, W), dtype=bool)            # cells whose state is known
        self.visited = np.zeros((H, W), dtype=bool)             # team-covered cells known to me

    # ---- updates ----------------------------------------------------------
    def observe(self, visible_mask: np.ndarray, pos) -> None:
        """Fold in a fresh sensor reading (bool HxW of currently visible cells)."""
        self.observed |= visible_mask
        r, c = int(pos[0]), int(pos[1])
        self.observed[r, c] = True
        self.visited[r, c] = True

    def merge(self, other: "Belief") -> None:
        """Graph-union another agent's belief into this one (share on contact)."""
        self.observed |= other.observed
        self.visited |= other.visited

    # ---- derived views ----------------------------------------------------
    @property
    def wall(self) -> np.ndarray:
        return self.observed & self._wall_truth

    @property
    def free(self) -> np.ndarray:
        return self.observed & ~self._wall_truth

    def frontiers(self) -> np.ndarray:
        """Known-free cells adjacent to an unobserved cell -> exploration targets."""
        unknown = ~self.observed
        up = np.zeros_like(unknown); up[:-1, :] = unknown[1:, :]
        dn = np.zeros_like(unknown); dn[1:, :] = unknown[:-1, :]
        lf = np.zeros_like(unknown); lf[:, :-1] = unknown[:, 1:]
        rt = np.zeros_like(unknown); rt[:, 1:] = unknown[:, :-1]
        adj_unknown = up | dn | lf | rt
        return self.free & adj_unknown
