"""
Shared environment for the three MARL tutorials (joint / CTDE / independent).

A cooperative coverage task with RANGE-LIMITED communication:

  * N agents on an H×W grid; each step every agent "explores" the cells within
    a vision radius of itself.
  * Agents GOSSIP their explored maps to neighbours within a comm radius, with
    a 1-step delay ("transmit what they explored last step"). Information floods
    multi-hop over time through whoever is connected — the comm graph is dynamic
    (it changes as agents move).
  * Per-agent observation is a 4-channel grid for a CNN; a separate global state
    is exposed for the centralized critics (CTDE / joint).
  * Reward is dense and common (cooperative): new team-explored cells per step.

This is a custom ``zymera.Env`` with its own JAX state pytree, so it threads
through ``zymera.rollout`` / ``jit`` / ``vmap`` unchanged. The three trainers
import :class:`CommCoverageEnv` and :func:`global_state`; only the
centralization of the actor/critic differs between them.
"""

from __future__ import annotations

import math
import os
import sys

import chex
import jax
import jax.numpy as jnp

import zymera

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cbf

# ---- defaults (the tutorials' setup) ---------------------------------------
# ---- REAL TASK (16×16) — the size-invariant approach validated on the dev world ----
GRID = 16
N_AGENTS = 4
MAX_STEPS = 70
VIS_R = 0          # EXPLORE/cover radius: 0 ⇒ only the cell the agent stands on is covered (1×1)
COMM_R = 5         # comm radius (Chebyshev): 5×5 partial comms
N_OBSTACLES = 0
SPAWN_RADIUS = 2   # agents start CLUSTERED within this Chebyshev radius of a random
                   # interior anchor (a 5×5 patch). None → scattered across the grid.

# Swarm reward: explore (individual discovery, raw cell counts) + stay connected
# (SMOOTH per-agent bonus, positive) − collisions (STRONG). No max-distance
# formation — the swarm may stretch the comm graph into a CHAIN (up to comm
# range per link) while staying connected, which lets it cover AND stay linked.
W_COV, W_CONN, W_COLL = 1.0, 2.0, 4.0
# Anti-redundancy: penalize an agent for covering cells ANOTHER agent covers the
# SAME step. Breaks the clump↔chain symmetry toward a SPREAD side-by-side sweep
# (division of labour) → lower redundancy → more unique coverage. 0 → off.
# NOT the cumulative-revisit penalty (that perversely discouraged covering ground);
# this only discourages doubling-up now, so it never says "leave a cell unexplored".
W_OVERLAP = 0.0
# Soft-CBF terms (OFF by default). When enabled, set W_CONN=W_COLL=0 so the CBF
# *replaces* the ad-hoc connectivity/collision terms with discrete-time barrier
# violation penalties. eps=0.1 calibrated: a stretched chain (λ₂≈0.3-0.5) is
# safe, a disconnected graph (λ₂≈0) violates; α=0.5 allows ~halving λ₂ per step
# (free spreading) while penalizing faster drops toward disconnection.
W_CBF_CONN, W_CBF_COLL = 0.0, 0.0
CBF_ALPHA, CBF_EPS, CBF_DMIN, CBF_SHARP = 0.5, 0.1, 1.0, 2.0
# Cohesion "leash": local freedom within COHESION_LEASH of the nearest teammate,
# soft penalty beyond → no agent fully leaves the herd (but the swarm may still
# stretch into a connected chain). 0 weight → off.
W_COHESION, COHESION_LEASH = 0.0, 4.0
# Degree floor: penalize an agent with fewer than DEGREE_FLOOR in-range neighbours
# (fully LOCAL — each agent counts its own in-range teammates; no λ₂, no global graph).
W_DEGREE, DEGREE_FLOOR = 0.0, 1.0
# Potential-based dense coverage shaping (PBRS, policy-invariant, size-invariant).
# F = γ·Φ(s') − Φ(s), added on top of the real reward — only densifies learning.
#  Φ1 (per-agent): −Cheby dist to nearest UNCOVERED cell (frontier-seeking).
#  Φ3 (team): −MEAN over uncovered cells of dist-to-nearest-agent (blanket/territory).
# Both built from absolute cell distances → per-step magnitude is grid-size-independent.
W_SHAPE1, W_SHAPE3, SHAPE_GAMMA = 0.0, 0.0, 0.99

# STAY, NORTH, EAST, SOUTH, WEST
_DELTAS = jnp.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32)


# ---- state -----------------------------------------------------------------


@chex.dataclass(frozen=True)
class CommState:
    pos:           chex.Array   # (N, 2) int32 — agent positions
    explored_by:   chex.Array   # (N, H, W) bool — each agent's own accumulated vision
    shared:        chex.Array   # (N, H, W) bool — own ∪ gossiped (1-step delayed)
    wall:          chex.Array   # (H, W) bool
    team_explored: chex.Array   # (H, W) bool — union of explored_by (ground truth)
    step_count:    chex.Array   # () int32
    visit_age:     chex.Array   # (N, H, W) float32 — steps since agent i last stood on each cell
                                #   (always carried; only exposed in obs when mem_channels=True)


# ---- geometry helpers (all pure JAX) ---------------------------------------


def _cheby_footprint(pos: chex.Array, h: int, w: int, r: int) -> chex.Array:
    """(N, H, W) bool — cells within Chebyshev radius ``r`` of each agent."""
    rows = jnp.arange(h)[None, :, None]
    cols = jnp.arange(w)[None, None, :]
    pr = pos[:, 0][:, None, None]
    pc = pos[:, 1][:, None, None]
    return jnp.maximum(jnp.abs(rows - pr), jnp.abs(cols - pc)) <= r


def _pairwise_cheby(pos: chex.Array) -> chex.Array:
    """(N, N) Chebyshev distances between agents."""
    return jnp.max(jnp.abs(pos[:, None, :] - pos[None, :, :]), axis=-1)


def _cell_cheby(pos: chex.Array, h: int, w: int) -> chex.Array:
    """(N, H, W) Chebyshev distance from each agent to every cell."""
    ys = jnp.arange(h)[None, :, None]
    xs = jnp.arange(w)[None, None, :]
    dr = jnp.abs(ys - pos[:, 0, None, None])
    dc = jnp.abs(xs - pos[:, 1, None, None])
    return jnp.maximum(dr, dc)


def _territory_owner(pos: chex.Array, comm_r: int, h: int, w: int) -> chex.Array:
    """(N, H, W) bool — agent i 'owns' a cell if it's the nearest IN-RANGE agent to it
    (partial-info Voronoi: out-of-range teammates don't compete; ties -> lowest index)."""
    dcell = _cell_cheby(pos, h, w)                                   # (N,H,W)
    inr = _adjacency(pos, comm_r)                                    # (N,N) in-range, diag True
    big = float(h + w + 1)
    dmask = jnp.where(inr[:, :, None, None], dcell[None], big)       # (N,N,H,W): i sees j only if in range
    owner = dmask.argmin(1)                                          # (N,H,W) nearest in-range agent (i's view)
    return owner == jnp.arange(pos.shape[0])[:, None, None]


def _adjacency(pos: chex.Array, r: int) -> chex.Array:
    """(N, N) bool — Chebyshev comm graph (symmetric, diagonal True)."""
    return _pairwise_cheby(pos) <= r


def _reach(adj: chex.Array) -> chex.Array:
    """(N, N) bool transitive closure — reach[i, j] = j reachable from i."""
    n = adj.shape[0]                       # static (number of agents)
    reach = adj
    for _ in range(math.ceil(math.log2(max(n, 2))) + 1):
        reach = reach | (jnp.matmul(reach.astype(jnp.int32), reach.astype(jnp.int32)) > 0)
    return reach


def _connected(adj: chex.Array) -> chex.Array:
    """Scalar bool — is the comm graph connected?"""
    return _reach(adj).all()


def _onehot_pos(pos: chex.Array, h: int, w: int) -> chex.Array:
    """(N, H, W) bool — a single True at each agent's cell."""
    n = pos.shape[0]
    grid = jnp.zeros((n, h, w), dtype=jnp.bool_)
    return grid.at[jnp.arange(n), pos[:, 0], pos[:, 1]].set(True)


def _move(pos: chex.Array, action: chex.Array, wall: chex.Array,
          h: int, w: int) -> chex.Array:
    """Apply actions; clip at the boundary and reject moves into walls."""
    proposed = pos + _DELTAS[action]
    r = jnp.clip(proposed[:, 0], 0, h - 1)
    c = jnp.clip(proposed[:, 1], 0, w - 1)
    hits = wall[r, c]
    r = jnp.where(hits, pos[:, 0], r)
    c = jnp.where(hits, pos[:, 1], c)
    return jnp.stack([r, c], axis=-1).astype(jnp.int32)


def _resolve_collisions(old_pos: chex.Array, target: chex.Array) -> chex.Array:
    """HARD collision avoidance: commit moves one agent at a time; an agent whose
    target cell is already taken by another agent reverts to its old cell. A
    lax.scan over agents (each commit visible to the next) → guarantees no two
    agents ever share a cell. Effectively masks colliding moves to a no-op."""
    n = old_pos.shape[0]

    def body(finalized, i):
        ti = target[i]
        same = jnp.all(finalized == ti[None, :], axis=-1)     # cells equal to my target
        same = same.at[i].set(False)                          # ignore self
        new_i = jnp.where(same.any(), old_pos[i], ti)         # taken → stay
        return finalized.at[i].set(new_i), None

    finalized, _ = jax.lax.scan(body, old_pos, jnp.arange(n))
    return finalized


def _nearest_uncovered_dist(pos: chex.Array, uncovered: chex.Array, h: int, w: int) -> chex.Array:
    """(N,) Chebyshev distance from each agent to its nearest UNCOVERED cell
    (0 if nothing uncovered). Φ1 = −this. Absolute cells → size-invariant."""
    rows = jnp.arange(h)[None, :, None]
    cols = jnp.arange(w)[None, None, :]
    ay = pos[:, 0][:, None, None]
    ax = pos[:, 1][:, None, None]
    d = jnp.maximum(jnp.abs(rows - ay), jnp.abs(cols - ax))        # (N, h, w)
    d_unc = jnp.where(uncovered[None], d, h + w + 1)
    md = d_unc.reshape(pos.shape[0], -1).min(-1)
    return jnp.where(uncovered.any(), md, 0.0).astype(jnp.float32)


def _nearest_own_uncovered_dist(pos: chex.Array, uncovered: chex.Array,
                                comm_r: int, h: int, w: int) -> chex.Array:
    """(N,) like _nearest_uncovered_dist but only over cells the agent OWNS (Voronoi).
    Anticipatory de-confliction: pulls each agent toward its OWN region's frontier.
    Falls back to the global nearest if the agent owns no uncovered cell."""
    owner = _territory_owner(pos, comm_r, h, w)                    # (N,h,w)
    dcell = _cell_cheby(pos, h, w)                                 # (N,h,w)
    n = pos.shape[0]
    big = float(h + w + 1)
    own_unc = uncovered[None] & owner                             # (N,h,w)
    own_d = jnp.where(own_unc, dcell, big).reshape(n, -1).min(-1)
    glob_d = jnp.where(uncovered[None], dcell, big).reshape(n, -1).min(-1)
    has_own = own_unc.reshape(n, -1).any(-1)
    return jnp.where(has_own, own_d, glob_d).astype(jnp.float32)


def _own_field_mean_dist(pos: chex.Array, uncovered: chex.Array,
                         comm_r: int, h: int, w: int) -> chex.Array:
    """(N,) MEAN over each agent's OWN-territory uncovered cells of distance to it.
    Looks at the WHOLE region (not just nearest), so the territory split is non-vacuous:
    minimizing it pulls each agent to cover its entire Voronoi share."""
    owner = _territory_owner(pos, comm_r, h, w)                    # (N,h,w)
    dcell = _cell_cheby(pos, h, w)                                 # (N,h,w)
    n = pos.shape[0]
    own_unc = (uncovered[None] & owner).astype(jnp.float32)
    s = (dcell * own_unc).reshape(n, -1).sum(-1)
    cnt = own_unc.reshape(n, -1).sum(-1)
    return jnp.where(cnt > 0, s / jnp.maximum(cnt, 1.0), 0.0).astype(jnp.float32)


def _field_mean_dist(pos: chex.Array, uncovered: chex.Array, h: int, w: int) -> chex.Array:
    """Scalar: MEAN over uncovered cells of Chebyshev distance to the nearest agent
    (0 if nothing uncovered). Φ3 = −this. Mean (not sum) → size-invariant."""
    rows = jnp.arange(h)[:, None, None]
    cols = jnp.arange(w)[None, :, None]
    ay = pos[:, 0][None, None, :]
    ax = pos[:, 1][None, None, :]
    nearest = jnp.maximum(jnp.abs(rows - ay), jnp.abs(cols - ax)).min(-1)   # (h, w)
    nunc = uncovered.sum()
    mean_d = jnp.where(uncovered, nearest, 0.0).sum() / jnp.maximum(nunc, 1)
    return jnp.where(nunc > 0, mean_d, 0.0).astype(jnp.float32)


def _random_wall(key, h, w, n_obstacles):
    if n_obstacles <= 0:
        return jnp.zeros((h, w), dtype=jnp.bool_)
    idx = jax.random.choice(key, h * w, shape=(n_obstacles,), replace=False)
    return jnp.zeros((h * w,), dtype=jnp.bool_).at[idx].set(True).reshape(h, w)


# ---- the env ---------------------------------------------------------------


class CommCoverageEnv(zymera.Env):
    """Cooperative coverage with range-limited gossip. Per-agent obs is a
    4-channel grid ``(N, 4, H, W)``; :func:`global_state` gives the centralized
    ``(3, H, W)`` view for the critics."""

    obs_channels = 5         # known · own-pos · known-walls · neighbours · local-frontier
    global_channels = 3

    def __init__(self, grid_h: int = GRID, grid_w: int = GRID,
                 n_agents: int = N_AGENTS, vis_r: int = VIS_R,
                 comm_r: int = COMM_R, n_obstacles: int = N_OBSTACLES,
                 sense_r: int = 1, spawn_radius: int = SPAWN_RADIUS,
                 w_cov: float = W_COV, w_conn: float = W_CONN,
                 w_coll: float = W_COLL, w_overlap: float = W_OVERLAP,
                 w_cbf_conn: float = W_CBF_CONN, w_cbf_coll: float = W_CBF_COLL,
                 cbf_alpha: float = CBF_ALPHA, cbf_eps: float = CBF_EPS,
                 cbf_dmin: float = CBF_DMIN, cbf_sharp: float = CBF_SHARP,
                 hard_collision: bool = False, conn_cap=None,
                 w_cohesion: float = W_COHESION, cohesion_leash: float = COHESION_LEASH,
                 w_degree: float = W_DEGREE, degree_floor: float = DEGREE_FLOOR,
                 w_shape1: float = W_SHAPE1, w_shape3: float = W_SHAPE3,
                 shape_gamma: float = SHAPE_GAMMA,
                 mem_channels: bool = False, w_revisit: float = 0.0,
                 recency_window: float = 16.0, territory_channel: bool = False,
                 terr_alpha: float = 1.0, terr_shaping: bool = False,
                 w_conn_barrier: float = 0.0, barrier_safe: float = 3.0,
                 barrier_sharp: float = 1.5, degree_channels: bool = False):
        super().__init__(n_agents=n_agents)
        self.H, self.W = grid_h, grid_w
        self.vis_r, self.comm_r = vis_r, comm_r
        self.n_obstacles = n_obstacles
        self.sense_r = sense_r           # local "where to move" sensing radius (Chebyshev)
        self.spawn_radius = spawn_radius  # cluster start within this radius (None → scatter)
        self.w_cov, self.w_conn = w_cov, w_conn       # reward weights (tunable per env
        self.w_coll, self.w_overlap = w_coll, w_overlap  # for the coverage-push sweep)
        self.w_cbf_conn, self.w_cbf_coll = w_cbf_conn, w_cbf_coll   # soft-CBF penalties
        self.cbf_alpha, self.cbf_eps = cbf_alpha, cbf_eps
        self.cbf_dmin, self.cbf_sharp = cbf_dmin, cbf_sharp
        self.hard_collision = hard_collision   # hard env-level collision masking (no two share a cell)
        self.conn_cap = conn_cap   # None → per-agent reachability; int → reward giant component capped here
                                   # (cap=N-1 ⇒ "3 connected + 1 roamer" scores like a full clump → no clump incentive)
        self.w_cohesion, self.cohesion_leash = w_cohesion, cohesion_leash   # soft tether to nearest teammate
        self.w_degree, self.degree_floor = w_degree, degree_floor   # LOCAL connectivity: penalize < floor in-range neighbours
        # MEMORY: expose per-agent own-trail + recency channels (obs 5 → 7); anti-revisit penalty
        self.mem_channels, self.w_revisit, self.recency_window = mem_channels, w_revisit, recency_window
        self.territory_channel = territory_channel   # per-agent "cells I'm the nearest in-range agent to"
        self.terr_alpha = terr_alpha   # coverage REWARD: own-territory cells pay full, teammates' pay alpha (1.0=off)
        self.terr_shaping = terr_shaping   # Φ1 pulls toward OWN-territory frontier (anticipatory de-confliction)
        # smooth LOCAL connectivity barrier: penalty ~0 while nearest teammate is inside
        # barrier_safe, ramps exp up to comm_r (capped). A graded degree-floor.
        self.w_conn_barrier, self.barrier_safe, self.barrier_sharp = w_conn_barrier, barrier_safe, barrier_sharp
        # DEGREE SIGNAL channels: broadcast planes of normalized degree (local
        # connectivity) + component fraction (gossip global-split awareness), so the
        # PPO policy can learn degree-driven role/return behavior (emergence).
        self.degree_channels = degree_channels
        self.obs_channels = (5 + (2 if mem_channels else 0) + (1 if territory_channel else 0)
                             + (2 if degree_channels else 0))
        self.w_shape1, self.w_shape3 = w_shape1, w_shape3   # PBRS dense-coverage shaping weights
        self.shape_gamma = shape_gamma

    # -- gym API ------------------------------------------------------------

    def _cluster_spawn(self, key, wall):
        """N distinct free cells clustered within ``spawn_radius`` of a random
        interior anchor. top_k over (block-and-free ≫ free ≫ wall) scores
        guarantees N distinct cells and degrades gracefully if the patch is
        crowded by obstacles (overflow spills to the nearest free cells)."""
        akey, ckey = jax.random.split(key)
        r = self.spawn_radius
        free = ~wall
        interior = jnp.zeros((self.H, self.W), bool).at[r:self.H - r, r:self.W - r].set(True)
        ap = (free & interior).reshape(-1).astype(jnp.float32)
        anchor = jax.random.choice(akey, self.H * self.W, p=ap / jnp.sum(ap))
        ar, ac = anchor // self.W, anchor % self.W
        rows = jnp.arange(self.H)[:, None]
        cols = jnp.arange(self.W)[None, :]
        block = (jnp.maximum(jnp.abs(rows - ar), jnp.abs(cols - ac)) <= r) & free
        noise = jax.random.uniform(ckey, (self.H * self.W,))
        score = jnp.where(block.reshape(-1), noise + 1.0,
                          jnp.where(free.reshape(-1), noise, -1.0))
        return jax.lax.top_k(score, self.n_agents)[1].astype(jnp.int32)

    def reset(self, key):
        wkey, skey = jax.random.split(key)
        wall = _random_wall(wkey, self.H, self.W, self.n_obstacles)
        if self.spawn_radius is None:
            free = (~wall).reshape(-1).astype(jnp.float32)
            idx = jax.random.choice(skey, self.H * self.W, shape=(self.n_agents,),
                                    replace=False, p=free / jnp.sum(free))
        else:
            idx = self._cluster_spawn(skey, wall)
        pos = jnp.stack([idx // self.W, idx % self.W], axis=-1).astype(jnp.int32)

        explored = _cheby_footprint(pos, self.H, self.W, self.vis_r) & ~wall[None]
        big = float(self.H + self.W)                                  # "never visited" sentinel age
        visit_age = jnp.where(_onehot_pos(pos, self.H, self.W), 0.0, big)
        state = CommState(
            pos=pos, explored_by=explored, shared=explored, wall=wall,
            team_explored=explored.any(0), step_count=jnp.zeros((), jnp.int32),
            visit_age=visit_age)
        return self._obs(state), state

    def step(self, state, action, key):
        del key
        new_pos = _move(state.pos, action, state.wall, self.H, self.W)
        if self.hard_collision:                       # hard mask: colliding moves → stay
            new_pos = _resolve_collisions(state.pos, new_pos)

        # own exploration (instant)
        vis = _cheby_footprint(new_pos, self.H, self.W, self.vis_r) & ~state.wall[None]
        explored = state.explored_by | vis

        # gossip: own ∪ neighbours' PREVIOUS shared map (1-step delay)
        adj = _adjacency(new_pos, self.comm_r)
        received = (adj[:, :, None, None] & state.shared[None, :, :, :]).any(axis=1)
        shared = explored | received

        team = explored.any(0)
        # per-agent recency: reset to 0 at my current cell, +1 everywhere else
        visit_age = jnp.where(_onehot_pos(new_pos, self.H, self.W), 0.0, state.visit_age + 1.0)
        new_state = CommState(
            pos=new_pos, explored_by=explored, shared=shared, wall=state.wall,
            team_explored=team, step_count=state.step_count + 1, visit_age=visit_age)

        # SWARM reward: explore (+) · stay connected (+, smooth) · spread / avoid
        # redundant overlap (−) · avoid collisions (−, strong)
        new_mask = vis & ~state.team_explored[None]                   # (N,H,W) cells I newly cover
        newly = new_mask.reshape(self.n_agents, -1).sum(-1)           # (N,) count
        # cooperation incentive: own-territory cells pay full, teammates' pay terr_alpha.
        # alpha=1.0 → identical to plain `newly` (off). Lower alpha → "stay in your lane".
        if self.terr_alpha < 1.0:
            owner = _territory_owner(new_pos, self.comm_r, self.H, self.W)   # (N,H,W)
            own_new = (new_mask & owner).reshape(self.n_agents, -1).sum(-1)
            other_new = (new_mask & ~owner).reshape(self.n_agents, -1).sum(-1)
            cover = own_new + self.terr_alpha * other_new
        else:
            cover = newly

        # anti-redundancy: cells I cover this step that ANOTHER agent also covers now
        seen_count = vis.sum(0)                                       # (H,W) agents seeing each cell
        others = (seen_count[None] - vis.astype(jnp.int32)) > 0       # (N,H,W) also seen by someone else
        overlap = (vis & others).reshape(self.n_agents, -1).sum(-1).astype(jnp.float32)  # (N,)

        d = _pairwise_cheby(new_pos)                                   # (N,N) Chebyshev
        off = ~jnp.eye(self.n_agents, dtype=bool)
        reach = _reach(d <= self.comm_r)                              # (N,N) closure
        if self.conn_cap is None:
            connectivity = (reach.sum(-1) - 1) / max(self.n_agents - 1, 1)  # (N,) per-agent reachability
        else:
            giant = reach.sum(-1).max()                               # largest connected component size
            connectivity = jnp.minimum(giant, self.conn_cap) / self.conn_cap  # capped → no clump bonus
        collisions = ((d == 0) & off).sum(-1).astype(jnp.float32)     # (N,) agents sharing my cell

        reward = (self.w_cov * cover + self.w_conn * connectivity
                  - self.w_overlap * overlap - self.w_coll * collisions)

        # anti-revisit: penalize stepping onto a cell the TEAM had already covered
        # BEFORE this step (keyed on state.team_explored → first-time coverage is never
        # penalized). Turns a wasted re-tread step from reward-0 into reward-negative.
        if self.w_revisit > 0.0:
            revisit = (vis & state.team_explored[None] & ~state.wall[None]
                       ).reshape(self.n_agents, -1).sum(-1).astype(jnp.float32)   # (N,)
            reward = reward - self.w_revisit * revisit

        # LOCAL connectivity (partial-info; no λ₂, no global graph) ----------------
        # (1) cohesion "leash": soft pull-back when the NEAREST teammate drifts past
        #     the leash. Capped at comm_r — an agent can't *measure* a teammate it
        #     can't sense, so beyond range this saturates (the degree floor below is
        #     what signals "I'm isolated"). Local freedom within the leash.
        if self.w_cohesion > 0.0:
            d_off = jnp.where(off, d, jnp.inf)
            nn = jnp.minimum(d_off.min(-1), float(self.comm_r))       # (N,) nearest, clamped to sensing
            reward = reward - self.w_cohesion * jnp.maximum(nn - self.cohesion_leash, 0.0)
        # (2) degree floor: each agent counts its OWN in-range teammates and is
        #     penalized for having fewer than `degree_floor` (floor=1 ⇒ "never fully
        #     isolate"). Purely local — the partial-info anti-fragmentation signal.
        if self.w_degree > 0.0:
            num_nb = ((d <= self.comm_r) & off).sum(-1).astype(jnp.float32)  # (N,) in-range neighbours
            reward = reward - self.w_degree * jnp.maximum(self.degree_floor - num_nb, 0.0)
        # (3) smooth connectivity BARRIER: exp ramp as the nearest teammate nears comm_r.
        #     ~0 inside barrier_safe, grows toward comm_r, CAPPED (nn clamped at comm_r) so
        #     it can't explode the gradient. A graded, gradient-bearing degree-floor.
        if self.w_conn_barrier > 0.0:
            d_off = jnp.where(off, d, jnp.inf)
            nn = jnp.minimum(d_off.min(-1), float(self.comm_r))       # (N,) nearest teammate, clamped
            barrier = jnp.exp(self.barrier_sharp * (nn - self.barrier_safe))  # small→large near comm_r
            reward = reward - self.w_conn_barrier * barrier

        # PBRS dense coverage shaping: F = γ·Φ(s') − Φ(s) (policy-invariant, size-invariant)
        if self.w_shape1 > 0.0 or self.w_shape3 > 0.0:
            g = self.shape_gamma
            unc0 = (~state.team_explored) & ~state.wall                # uncovered at s
            unc1 = (~team) & ~state.wall                               # uncovered at s'
            if self.w_shape1 > 0.0:                                    # Φ1: per-agent frontier-seeking
                if self.terr_shaping:        # pull toward OWN-territory frontier (de-confliction)
                    p0 = _nearest_own_uncovered_dist(state.pos, unc0, self.comm_r, self.H, self.W)
                    p1 = _nearest_own_uncovered_dist(new_pos, unc1, self.comm_r, self.H, self.W)
                else:
                    p0 = _nearest_uncovered_dist(state.pos, unc0, self.H, self.W)
                    p1 = _nearest_uncovered_dist(new_pos, unc1, self.H, self.W)
                reward = reward + self.w_shape1 * (g * (-p1) - (-p0))  # (N,)
            if self.w_shape3 > 0.0:                                    # Φ3: blanket field
                if self.terr_shaping:        # each agent covers its OWN region's whole field
                    q0 = _own_field_mean_dist(state.pos, unc0, self.comm_r, self.H, self.W)
                    q1 = _own_field_mean_dist(new_pos, unc1, self.comm_r, self.H, self.W)
                else:
                    q0 = _field_mean_dist(state.pos, unc0, self.H, self.W)
                    q1 = _field_mean_dist(new_pos, unc1, self.H, self.W)
                reward = reward + self.w_shape3 * (g * (-q1) - (-q0))

        # soft-CBF: discrete-time barrier violation residuals (x_k → x_{k+1}).
        # Only "bites" near a constraint boundary; a stretched connected chain
        # incurs ~0. Enabled per env (default off); gated at trace time so the
        # λ₂ eigendecomposition only compiles in when actually used.
        if self.w_cbf_conn > 0.0:
            hp = cbf.conn_barrier(state.pos, self.comm_r, self.cbf_eps, self.cbf_sharp)
            hn = cbf.conn_barrier(new_pos, self.comm_r, self.cbf_eps, self.cbf_sharp)
            reward = reward - self.w_cbf_conn * (cbf.residual(hp, hn, self.cbf_alpha)
                                                 / self.n_agents)     # shared team penalty
        if self.w_cbf_coll > 0.0:
            hp = cbf.coll_barriers(state.pos, self.cbf_dmin)          # (N,N)
            hn = cbf.coll_barriers(new_pos, self.cbf_dmin)
            reward = reward - self.w_cbf_coll * cbf.residual(hp, hn, self.cbf_alpha).sum(-1)

        reward = reward.astype(jnp.float32)                           # (N,) per agent
        done = jnp.zeros((self.n_agents,), dtype=jnp.bool_)
        return self._obs(new_state), new_state, reward, done, {}

    # -- observations -------------------------------------------------------

    def _obs(self, state):
        """Per-agent 4-channel grid: [known-explored, own-pos, known-walls,
        neighbour-positions]. Shape (N, 4, H, W)."""
        known = state.shared.astype(jnp.float32)
        own = _onehot_pos(state.pos, self.H, self.W).astype(jnp.float32)
        known_walls = (state.wall[None] & state.shared).astype(jnp.float32)

        adj = _adjacency(state.pos, self.comm_r) & ~jnp.eye(self.n_agents, dtype=bool)
        onehot = _onehot_pos(state.pos, self.H, self.W)
        neigh = (adj[:, :, None, None] & onehot[None, :, :, :]).any(axis=1).astype(jnp.float32)
        # local sense: cells the agent does NOT yet know, within sense_r of it →
        # a direct "which way is fresh ground nearby" signal (lowers policy entropy)
        window = _cheby_footprint(state.pos, self.H, self.W, self.sense_r)
        local_frontier = ((~state.shared) & window).astype(jnp.float32)
        chans = [known, own, known_walls, neigh, local_frontier]
        if self.degree_channels:
            # the degree SIGNAL as broadcast planes: normalized degree (# in-range
            # neighbours / (N-1)) and component fraction (|reachable|/N, the gossiped
            # global-split awareness). Lets the policy self-regulate connectivity.
            degree = adj.sum(-1).astype(jnp.float32) / max(self.n_agents - 1, 1)        # (N,)
            reach = _reach(_adjacency(state.pos, self.comm_r))
            comp_frac = reach.sum(-1).astype(jnp.float32) / self.n_agents               # (N,)
            ones = jnp.ones((self.n_agents, self.H, self.W), jnp.float32)
            chans += [degree[:, None, None] * ones, comp_frac[:, None, None] * ones]
        if self.territory_channel:
            # per-agent "my share": uncovered cells where I'm the nearest IN-RANGE agent.
            mine = _territory_owner(state.pos, self.comm_r, self.H, self.W)  # (N,H,W)
            unc = (~state.shared) & ~state.wall[None]                   # my uncovered-known cells
            chans.append((mine & unc).astype(jnp.float32))
        if self.mem_channels:
            # own-trail: cells I MYSELF have visited (vs `known` = team∪gossip) → "did *I* cover this?"
            own_trail = state.explored_by.astype(jnp.float32)                       # (N,H,W)
            # recency: how stale is each cell to me (0 = just left, 1 = long ago / never)
            recency = jnp.clip(state.visit_age, 0.0, self.recency_window) / self.recency_window
            chans += [own_trail, recency]
        return jnp.stack(chans, axis=1)

    def global_state(self, state):
        """Centralized (3, H, W) view for the critics: [team-explored,
        all-positions, walls]."""
        team = state.team_explored.astype(jnp.float32)
        allp = _onehot_pos(state.pos, self.H, self.W).any(0).astype(jnp.float32)
        walls = state.wall.astype(jnp.float32)
        return jnp.stack([team, allp, walls], axis=0)


def global_state(env: CommCoverageEnv, state) -> chex.Array:
    """``global_state`` over a (possibly stacked) state pytree — handles a
    trajectory by leaving the batch/time axes to ``jax.vmap`` in the trainer."""
    return env.global_state(state)


def make_env(**kw) -> CommCoverageEnv:
    return CommCoverageEnv(**kw)


zymera.register_env("comm-coverage-v0", CommCoverageEnv)


# ---- self-test -------------------------------------------------------------


def _selftest():
    env = make_env()
    key = jax.random.PRNGKey(0)

    def random_policy(obs, k):
        return jax.random.randint(k, (obs.shape[0],), 0, zymera.N_ACTIONS).astype(jnp.int32)

    traj = zymera.rollout(env, random_policy, MAX_STEPS, key)
    obs = traj["obs"]                      # (T+1, N, 4, H, W)
    states = traj["world"]
    cov = (states.team_explored.reshape(states.team_explored.shape[0], -1).sum(-1)
           / (GRID * GRID))                # (T+1,) coverage fraction
    # gossip sanity: shared should be a superset of own explored
    superset = bool((states.shared >= states.explored_by).all())
    print(f"obs shape           {tuple(obs.shape)}   (T+1, N, C, H, W)")
    print(f"global_state shape  {tuple(env.global_state(jax.tree_util.tree_map(lambda x: x[0], states)).shape)}   (C, H, W)")
    print(f"reward range        [{float(traj['reward'].min()):.4f}, {float(traj['reward'].max()):.4f}]  (dense, ≥0)")
    print(f"coverage  start {float(cov[0])*100:.0f}%  →  end {float(cov[-1])*100:.0f}%   (random policy, {MAX_STEPS} steps)")
    print(f"gossip superset-of-own invariant holds: {superset}")
    # vmap over seeds works (pure-JAX env)
    keys = jax.random.split(key, 8)
    batch = jax.vmap(lambda k: zymera.rollout(env, random_policy, MAX_STEPS, k))(keys)
    print(f"vmap 8 seeds → obs {tuple(batch['obs'].shape)}")


if __name__ == "__main__":
    _selftest()
