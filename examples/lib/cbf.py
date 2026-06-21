"""
Control-Barrier-Function terms for the comm-graph min/max constraints, used as
SOFT penalties (discrete-time CBF violation residuals) — no hard masking.

Safe sets (forward-invariant if the barrier never goes negative):
  collision:    h_coll_ij = d_ij - d_min               (≥0 : agents not overlapping)
  connectivity: h_conn     = λ₂(L_w(x)) - eps           (≥0 : graph connected w/ margin)

where λ₂ is the Fiedler value (2nd-smallest eigenvalue) of the SMOOTH-weighted comm
Laplacian — smooth in positions so the barrier has meaningful gradients.

Discrete-time CBF: a transition x_k → x_{k+1} is safe if
    h(x_{k+1}) ≥ (1 - α) · h(x_k),     0 < α ≤ 1
so we penalize the violation residual  relu((1-α)·h(x_k) - h(x_{k+1})).

Key property vs a maximize-connectivity reward: the barrier is a THRESHOLD
(λ₂ ≥ eps), so a just-connected stretched chain SATISFIES it — there is no
incentive to clump. Coverage is then free to stretch the swarm to the boundary.
"""

import jax
import jax.numpy as jnp


def _cheby(pos):
    """(N,N) Chebyshev distances between agents (float)."""
    return jnp.max(jnp.abs(pos[:, None, :] - pos[None, :, :]), -1).astype(jnp.float32)


def soft_weights(pos, comm_r, sharp=2.0):
    """(N,N) smooth edge weights ≈1 within comm_r, ≈0 beyond; zero diagonal."""
    d = _cheby(pos)
    w = jax.nn.sigmoid(sharp * (comm_r - d))          # smooth cutoff at the comm range
    n = pos.shape[0]
    return w * (1.0 - jnp.eye(n))


def lambda2(pos, comm_r, sharp=2.0):
    """Fiedler value (2nd-smallest eigenvalue) of the weighted Laplacian.
    λ₂ > 0 ⟺ the (weighted) graph is connected."""
    W = soft_weights(pos, comm_r, sharp)
    L = jnp.diag(W.sum(-1)) - W
    return jnp.linalg.eigvalsh(L)[1]                  # eigvalsh returns ascending order


def conn_barrier(pos, comm_r, eps, sharp=2.0):
    """Scalar connectivity barrier h_conn = λ₂ - eps."""
    return lambda2(pos, comm_r, sharp) - eps


def coll_barriers(pos, d_min=1.0):
    """(N,N) per-pair collision barrier h_ij = d_ij - d_min; self-pairs masked
    to a large value so they never register a violation."""
    d = _cheby(pos)
    n = pos.shape[0]
    return (d - d_min) + jnp.eye(n) * 1e3


def residual(h_prev, h_next, alpha):
    """Discrete-time CBF violation residual: relu((1-α)·h_prev - h_next)."""
    return jax.nn.relu((1.0 - alpha) * h_prev - h_next)
