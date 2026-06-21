"""Coordinated boustrophedon strip-sweep -- a coverage controller for high
*visited* (stepped-on) coverage by a connected swarm.

Rows are split into `n` horizontal strips (agent k owns rows [k*S, (k+1)*S)).
Every agent targets the *same column* each step (column-lockstep), so the swarm
forms a connected vertical chain that rakes left-right; advancing the row-offset
within each strip after each pass serpentines the whole grid. BFS routes around
obstacles; the schedule advances once the line has (mostly) reached the column,
with a small max-dwell so one stuck agent never stalls the sweep.

This is an *engineered* coverage strategy (a strong baseline / feasibility
demo), distinct from the learned/emergent compass. It exists to hit the
">90% visited, connected" target and to bound what good coverage looks like.
"""

from __future__ import annotations

import numpy as np

from .controller import action_for, bfs, first_step


class SweepController:
    def __init__(self, H: int, W: int, n: int, max_dwell: int = 12):
        self.H, self.W, self.n = H, W, n
        self.S = max(1, H // n)                 # rows per strip
        self.max_dwell = max_dwell
        self.max_off = self.S + (1 if H % n else 0)
        self.schedule = []                      # serpentine (offset, column)
        for off in range(self.max_off):
            cols = range(W) if off % 2 == 0 else range(W - 1, -1, -1)
            for col in cols:
                self.schedule.append((off, col))
        self.idx = 0
        self.strip_of = None                    # agent -> strip, assigned by start row

    def _assign(self, positions):
        order = np.argsort(np.asarray(positions)[:, 0])   # agents sorted top->bottom
        self.strip_of = {int(k): s for s, k in enumerate(order)}

    def _strip_bounds(self, strip):
        r0 = strip * self.S
        r1 = self.H if strip == self.n - 1 else (strip + 1) * self.S  # last strip takes remainder
        return r0, r1

    def _target(self, k, off, col, wall):
        r0, r1 = self._strip_bounds(self.strip_of[k])
        tr = min(r0 + off, r1 - 1)
        if wall[tr, col]:                       # snap to nearest free row in this strip column
            cand = [r for r in range(r0, r1) if not wall[r, col]]
            if not cand:
                return None
            tr = min(cand, key=lambda r: abs(r - (r0 + off)))
        return (tr, col)

    def actions(self, positions, wall):
        if self.strip_of is None:
            self._assign(positions)
        off, col = self.schedule[min(self.idx, len(self.schedule) - 1)]
        free = ~wall
        acts = np.zeros(self.n, dtype=int)
        ready = 0                               # at / one step from / unable to reach target
        for k in range(self.n):
            t = self._target(k, off, col, wall)
            src = (int(positions[k][0]), int(positions[k][1]))
            if t is None:
                ready += 1
                continue
            if src != t:
                dist, parent = bfs(free, src)
                if not np.isfinite(dist[t[0], t[1]]):
                    ready += 1                  # unreachable -> don't stall the sweep
                    continue
                acts[k] = action_for(src, first_step(parent, src, t))
            if abs(src[0] - t[0]) + abs(src[1] - t[1]) <= 1:
                ready += 1                      # will reach this tick
        # advance one column per step when the line is in lockstep (within one
        # step of the target); wait when an agent is detouring around obstacles.
        if ready >= self.n - 1:
            self.idx = min(self.idx + 1, len(self.schedule) - 1)
        return acts
