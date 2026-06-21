"""
Independent PPO (IPPO) — the fully DECENTRALIZED point of the trilogy: every
agent has its own actor AND its own critic, both from its own local obs.
Parameter-shared across agents. (MARL book: independent learning.)

    actor  π(a_i | o_i)   ← local obs
    critic V(o_i)         ← local obs
"""

import sys
from pathlib import Path

import equinox as eqx
import jax

sys.path.insert(0, str(Path(__file__).parent))
import _marl_core as core
import comm_coverage as cc


class IndependentAC(eqx.Module):
    actor_enc: core.Encoder
    actor_head: eqx.nn.Linear
    critic_enc: core.Encoder
    critic_head: eqx.nn.Linear

    def __init__(self, env, key):
        ka, kah, kc, kch = jax.random.split(key, 4)
        ch = env.obs_channels
        self.actor_enc = core.Encoder(ch, ka)
        self.actor_head = eqx.nn.Linear(128, core.N_ACTIONS, key=kah)
        self.critic_enc = core.Encoder(ch, kc)
        self.critic_head = eqx.nn.Linear(128, 1, key=kch)

    def logits(self, obs):                 # (N,C,H,W) -> (N,A)
        return jax.vmap(lambda o: self.actor_head(self.actor_enc(o)))(obs)

    def values(self, obs, gstate):         # local critic; ignores the global state
        del gstate
        return jax.vmap(lambda o: self.critic_head(self.critic_enc(o))[0])(obs)


if __name__ == "__main__":
    core.run(cc.make_env(), IndependentAC, "IPPO (independent)")
