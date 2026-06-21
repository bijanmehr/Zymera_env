"""The learned role policy (v2, evolution-trained).

A tiny shared MLP that maps an agent's LOCAL features -> its own connectivity
weight `lambda_self` (the relay/frontier role, continuous). The goal-selection
compass is reused unchanged; the policy only learns *how much each agent should
prioritize holding the network vs. exploring*, from where it sits in the swarm.
Gradient-free (trained by ES in evolve.py), so it stays pure numpy and plugs
into the existing host-side sim with no JAX rewrite.
"""
from __future__ import annotations

import numpy as np

FEAT_DIM = 6
HIDDEN = 16
LAM_MAX = 10.0
_SHAPES = [(FEAT_DIM, HIDDEN), (HIDDEN,), (HIDDEN, 1), (1,)]
N_PARAMS = sum(int(np.prod(s)) for s in _SHAPES)


def init_theta(seed=0):
    rng = np.random.default_rng(seed)
    parts = [rng.normal(0, 0.5, _SHAPES[0]), np.zeros(_SHAPES[1]),
             rng.normal(0, 0.5, _SHAPES[2]), np.zeros(_SHAPES[3])]
    return np.concatenate([p.ravel() for p in parts])


def _unflatten(theta):
    out, k = [], 0
    for s in _SHAPES:
        n = int(np.prod(s))
        out.append(theta[k:k + n].reshape(s))
        k += n
    return out


def policy_lambda(theta, feats):
    """feats (FEAT_DIM,) -> lambda_self in [0, LAM_MAX]."""
    W1, b1, W2, b2 = _unflatten(theta)
    h = np.tanh(feats @ W1 + b1)
    o = float((h @ W2 + b2)[0])
    return LAM_MAX / (1.0 + np.exp(-o))


# ---- relay-gate policy (v3): learns WHEN to relay, not how much to weight ----
# The lambda head had no leverage at scale (flat ES). The relay GATE flips an
# agent between frontier-seeking and active bridge-holding -- a large-effect
# decision, so ES has a real gradient to climb. Features are the 6 local ones
# plus 2 from connect.relay_features (global-cut-vertex, edge pressure).
GATE_FEAT_DIM = 8
GATE_HIDDEN = 16
_GSHAPES = [(GATE_FEAT_DIM, GATE_HIDDEN), (GATE_HIDDEN,), (GATE_HIDDEN, 1), (1,)]
N_GATE_PARAMS = sum(int(np.prod(s)) for s in _GSHAPES)


def init_gate_theta(seed=0):
    rng = np.random.default_rng(seed)
    parts = [rng.normal(0, 0.5, _GSHAPES[0]), np.zeros(_GSHAPES[1]),
             rng.normal(0, 0.5, _GSHAPES[2]), np.zeros(_GSHAPES[3])]
    return np.concatenate([p.ravel() for p in parts])


def _gunflatten(theta):
    out, k = [], 0
    for s in _GSHAPES:
        n = int(np.prod(s))
        out.append(theta[k:k + n].reshape(s))
        k += n
    return out


def relay_gate(theta, feats):
    """feats (GATE_FEAT_DIM,) -> P(relay) in [0, 1]. >0.5 => take the relay role."""
    W1, b1, W2, b2 = _gunflatten(theta)
    h = np.tanh(feats @ W1 + b1)
    o = float((h @ W2 + b2)[0])
    return 1.0 / (1.0 + np.exp(-o))


# ---- degree-signal policy (v4): learns WHEN to fall back from the degree budget ----
# Instead of a hand-set degree floor, the policy maps degree-based features (incl.
# the gossiped component fraction = global-split awareness) -> P(return to herd).
# ES/MARL discover the role-switch rule, so relay/frontier specialization can EMERGE.
DEG_FEAT_DIM = 6
DEG_HIDDEN = 16
_DSHAPES = [(DEG_FEAT_DIM, DEG_HIDDEN), (DEG_HIDDEN,), (DEG_HIDDEN, 1), (1,)]
N_DEG_PARAMS = sum(int(np.prod(s)) for s in _DSHAPES)


def init_degree_theta(seed=0):
    rng = np.random.default_rng(seed)
    parts = [rng.normal(0, 0.5, _DSHAPES[0]), np.zeros(_DSHAPES[1]),
             rng.normal(0, 0.5, _DSHAPES[2]), np.zeros(_DSHAPES[3])]
    return np.concatenate([p.ravel() for p in parts])


def _dunflatten(theta):
    out, k = [], 0
    for s in _DSHAPES:
        n = int(np.prod(s))
        out.append(theta[k:k + n].reshape(s))
        k += n
    return out


def degree_policy(theta, feats):
    """feats (DEG_FEAT_DIM,) -> P(return-to-herd) in [0, 1]. >0.5 => fall back."""
    W1, b1, W2, b2 = _dunflatten(theta)
    h = np.tanh(feats @ W1 + b1)
    o = float((h @ W2 + b2)[0])
    return 1.0 / (1.0 + np.exp(-o))


# ============ v1 — GNN role head (1-round message passing over the comm graph) ============
# Refines the backbone's tree role: a leaf may be FLIPPED to relay if its graph
# context (own features + aggregated neighbor features) says it's load-bearing.
# Permutation-invariant (mean aggregate) and variable-N by construction.
GNN_FEAT = 6                       # the 6 local features from swarm._agent_features
GNN_IN = 2 * GNN_FEAT              # own || mean(neighbors)
GNN_HID = 16
_GGSHAPES = [(GNN_IN, GNN_HID), (GNN_HID,), (GNN_HID, 1), (1,)]
N_GNN_PARAMS = sum(int(np.prod(s)) for s in _GGSHAPES)


def init_gnn_theta(seed=0):
    rng = np.random.default_rng(seed)
    parts = [rng.normal(0, 0.5, _GGSHAPES[0]), np.zeros(_GGSHAPES[1]),
             rng.normal(0, 0.5, _GGSHAPES[2]), np.zeros(_GGSHAPES[3])]
    return np.concatenate([p.ravel() for p in parts])


def _gg_unflatten(theta):
    out, k = [], 0
    for s in _GGSHAPES:
        n = int(np.prod(s))
        out.append(theta[k:k + n].reshape(s)); k += n
    return out


def gnn_relay(theta, feats, nbr_feats):
    """feats (6,), nbr_feats list of (6,) -> P(act as relay) in [0,1]."""
    agg = np.mean(np.stack(nbr_feats), 0) if nbr_feats else np.zeros_like(feats)
    x = np.concatenate([feats, agg])
    W1, b1, W2, b2 = _gg_unflatten(theta)
    h = np.tanh(x @ W1 + b1)
    o = float((h @ W2 + b2)[0])
    return 1.0 / (1.0 + np.exp(-o))


# ============ v2 — attention goal head (learned ranking over candidate cells) ============
# A leaf scores its reachable known-free candidate cells and picks the best — a
# learned pointer over a variable candidate set (the size-invariant goal head).
# Per-candidate features: info-gain, closeness, parent-range margin, frontier-richness.
ATTN_IN = 4
ATTN_HID = 8
_AASHAPES = [(ATTN_IN, ATTN_HID), (ATTN_HID,), (ATTN_HID, 1), (1,)]
N_ATTN_PARAMS = sum(int(np.prod(s)) for s in _AASHAPES)


def init_attn_theta(seed=0):
    rng = np.random.default_rng(seed)
    parts = [rng.normal(0, 0.5, _AASHAPES[0]), np.zeros(_AASHAPES[1]),
             rng.normal(0, 0.5, _AASHAPES[2]), np.zeros(_AASHAPES[3])]
    return np.concatenate([p.ravel() for p in parts])


def _aa_unflatten(theta):
    out, k = [], 0
    for s in _AASHAPES:
        n = int(np.prod(s))
        out.append(theta[k:k + n].reshape(s)); k += n
    return out


def attn_goal(theta, i, pos, parent, kb_count, seen, wall, dist, comm_r):
    """Learned ranking over candidate frontier cells -> chosen goal (r,c)."""
    free = seen & ~wall
    cand = np.argwhere(free & np.isfinite(dist))
    if len(cand) == 0:
        return (int(pos[i, 0]), int(pos[i, 1]))
    r, c = cand[:, 0], cand[:, 1]
    gain = 1.0 / np.sqrt(kb_count[r, c] + 1.0)                       # info-gain
    d = dist[r, c]; close = 1.0 - d / (d.max() + 1e-6)               # closeness
    if parent[i] >= 0:
        pr, pc = int(pos[parent[i], 0]), int(pos[parent[i], 1])
        cheb = np.maximum(np.abs(r - pr), np.abs(c - pc))
        margin = (comm_r - cheb) / comm_r                            # +ve = safely linked
    else:
        margin = np.ones(len(cand))
    # frontier-richness: how many 4-neighbors of the candidate are still unknown
    H, W = seen.shape
    unk = (~seen).astype(np.float32)
    rich = np.zeros(len(cand), np.float32)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = np.clip(r + dr, 0, H - 1); cc = np.clip(c + dc, 0, W - 1)
        rich += unk[rr, cc]
    rich /= 4.0
    X = np.stack([gain, close, margin, rich], 1)                     # (M,4)
    W1, b1, W2, b2 = _aa_unflatten(theta)
    h = np.tanh(X @ W1 + b1)
    s = (h @ W2 + b2)[:, 0]
    idx = int(np.argmax(s))
    return (int(r[idx]), int(c[idx]))
