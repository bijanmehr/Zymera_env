"""
CTDE PPO (MAPPO-style) — the MIDDLE of the trilogy: decentralized actors, one
CENTRALIZED critic that sees the global state (used only at training time; at
execution the actors run from local obs alone). (MARL book Ch 9: CTDE.)

    actor  π(a_i | o_i)   ← local obs        (decentralized)
    critic V(s)           ← global state     (centralized; one team value)
"""

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))
import _marl_core as core
import comm_coverage as cc


class CTDE_AC(eqx.Module):
    actor_enc: core.Encoder
    actor_head: eqx.nn.Linear
    critic_enc: core.Encoder       # encodes the GLOBAL (3,H,W) state
    critic_head: eqx.nn.Linear

    def __init__(self, env, key):
        ka, kah, kc, kch = jax.random.split(key, 4)
        self.actor_enc = core.Encoder(env.obs_channels, ka)
        self.actor_head = eqx.nn.Linear(128, core.N_ACTIONS, key=kah)
        self.critic_enc = core.Encoder(env.global_channels, kc)
        self.critic_head = eqx.nn.Linear(128, 1, key=kch)

    def logits(self, obs):                 # (N,C,H,W) -> (N,A)  (decentralized)
        return jax.vmap(lambda o: self.actor_head(self.actor_enc(o)))(obs)

    def values(self, obs, gstate):         # centralized: one team value, broadcast to N
        v = self.critic_head(self.critic_enc(gstate))[0]
        return jnp.full((obs.shape[0],), v)


if __name__ == "__main__":
    core.run(cc.make_env(), CTDE_AC, "MAPPO (CTDE · centralized critic)")
