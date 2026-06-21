"""
Joint PPO — the fully CENTRALIZED point of the trilogy: one network sees the
whole team's observation and emits all N agents' actions; one centralized
critic. The team is effectively a single agent over the joint action space.
(MARL book: joint / central learning.)

    actor  π(a_1..a_N | o_1..o_N)   ← all agents' obs (centralized)
    critic V(s)                     ← global state    (centralized)
"""

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))
import _marl_core as core
import comm_coverage as cc


class JointAC(eqx.Module):
    enc: core.Encoder              # shared per-agent feature encoder
    jh1: eqx.nn.Linear             # joint head (plain Linears — no stored activation
    jh2: eqx.nn.Linear             # function, which lax.scan can't carry)
    critic_enc: core.Encoder
    critic_head: eqx.nn.Linear
    n_agents: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)

    def __init__(self, env, key):
        ke, k1, k2, kc, kch = jax.random.split(key, 5)
        self.n_agents = env.n_agents
        self.n_actions = core.N_ACTIONS
        self.enc = core.Encoder(env.obs_channels, ke)
        self.jh1 = eqx.nn.Linear(128 * env.n_agents, 128, key=k1)
        self.jh2 = eqx.nn.Linear(128, core.N_ACTIONS * env.n_agents, key=k2)
        self.critic_enc = core.Encoder(env.global_channels, kc)
        self.critic_head = eqx.nn.Linear(128, 1, key=kch)

    def logits(self, obs):                 # (N,C,H,W) -> (N,A), jointly conditioned
        feats = jax.vmap(self.enc)(obs)                  # (N,128)
        h = jax.nn.relu(self.jh1(feats.reshape(-1)))     # joint context over all agents
        return self.jh2(h).reshape(self.n_agents, self.n_actions)

    def values(self, obs, gstate):         # centralized team value, broadcast to N
        v = self.critic_head(self.critic_enc(gstate))[0]
        return jnp.full((obs.shape[0],), v)


if __name__ == "__main__":
    core.run(cc.make_env(), JointAC, "Joint PPO (fully centralized)")
