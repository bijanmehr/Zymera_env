"""
``Helper`` — the in-between layer between world physics and agent code.

Owns utilities that aren't purely env state and aren't purely policy
state: communication graph, termination rules, observation augmentation.
Today only ``comm`` does real work; ``terminate`` / ``augment`` ship as
identity placeholders so the contract is committed before consumers
depend on it.

::

    from zymera import Helper
    h = Helper(comm_radius=2)
    adj = h.comm(world)
"""

import chex
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class Helper:
    """Comm / terminate / augment service.

    ``chex.dataclass`` pytree — threads through ``jax.jit`` / ``vmap``.
    The only field today is ``comm_radius``; methods are pure JAX with
    no Python branching on traced values.
    """

    comm_radius: int

    # ---- comms ---------------------------------------------------------

    def comm(self, world) -> chex.Array:
        """Pairwise comm adjacency for agents in ``world``.

        Returns ``(N, N)`` bool — True where two agents are within
        ``comm_radius`` Chebyshev steps. Symmetric, diagonal True.
        """
        pos = world.body.position
        diff = pos[:, None, :] - pos[None, :, :]            # (N, N, 2)
        dist = jnp.max(jnp.abs(diff), axis=-1)              # (N, N) Chebyshev
        return dist <= self.comm_radius

    # ---- placeholders --------------------------------------------------

    def terminate(self, world, t: chex.Array) -> chex.Array:
        """Per-agent termination flag. Placeholder: all-False.

        Signature is the real one; real termination rules (timeouts,
        goal-reached, mission-failed) slot in by subclassing.
        """
        n = world.body.position.shape[0]
        return jnp.zeros((n,), dtype=jnp.bool_)

    def augment(self, obs: chex.Array, world) -> chex.Array:
        """Decorate raw obs before the policy sees it. Placeholder: identity.

        Real augmentations (neighbour pooling through comm_graph, mission
        cues, local-window slices) replace this method later.
        """
        return obs
