"""The Compass -- goal selection over the belief.

v1 is a deterministic scorer (the learned policy is the planned replacement):
each reachable frontier cell is scored as

    score = w_gain * (unknown cells it reveals)
          - w_cost * (path distance to it)
          - w_crowd * (already-covered cells nearby)        [anti-crowding]
          - w_claim * (1 if a teammate has claimed near it) [anti-crowding]

The two anti-crowding terms are the de-confliction signal the literature
identifies as the load-bearing ingredient for emergent division of labor
(turning shared-map exploration from the "herd effect" into spatial
partitioning). Toggle them off to reproduce the herd baseline.
"""

from __future__ import annotations

import numpy as np


def _window_count(mask: np.ndarray, r: int, c: int, rad: int) -> int:
    H, W = mask.shape
    r0, r1 = max(0, r - rad), min(H, r + rad + 1)
    c0, c1 = max(0, c - rad), min(W, c + rad + 1)
    return int(mask[r0:r1, c0:c1].sum())


def choose_goal(belief, src, dist_map, claims, *, anti_crowding=True,
                anchor=None, tether_radius=None, gain_radius=2, crowd_radius=3,
                claim_radius=5, w_gain=1.0, w_cost=0.15, w_crowd=1.5, w_anchor=0.0):
    """Pick the best reachable frontier cell. Returns (r, c) or None.

    With anti_crowding, a frontier within `claim_radius` of a teammate's claimed
    goal is *hard-excluded* (decisive de-confliction -> agents take different
    regions), falling back to the best overall only if every frontier is claimed.

    `anchor` (+ w_anchor) pulls every agent toward frontiers near a shared point
    (the swarm centroid) so a *connected* group migrates coherently instead of
    being torn apart by each agent chasing its own far frontier.
    """
    frontier = belief.frontiers()
    cells = np.argwhere(frontier)
    if len(cells) == 0:
        return None

    unknown = ~belief.observed
    scored = []                                    # (score, excluded, (r, c))
    for r, c in cells:
        d = dist_map[r, c]
        if not np.isfinite(d):
            continue
        score = w_gain * _window_count(unknown, r, c, gain_radius) - w_cost * d
        if anchor is not None:
            score -= w_anchor * max(abs(r - anchor[0]), abs(c - anchor[1]))
        excluded = False
        if anchor is not None and tether_radius is not None and \
                max(abs(r - anchor[0]), abs(c - anchor[1])) > tether_radius:
            excluded = True                    # keep goals near the swarm (cohesion)
        if anti_crowding:
            score -= w_crowd * _window_count(belief.visited, r, c, crowd_radius)
            for cr, cc in claims:
                if abs(cr - r) <= claim_radius and abs(cc - c) <= claim_radius:
                    excluded = True
                    break
        scored.append((score, excluded, (int(r), int(c))))

    if not scored:
        return None
    free = [s for s in scored if not s[1]]
    pool = free if free else scored                # prefer de-conflicted targets
    return max(pool, key=lambda x: x[0])[2]
