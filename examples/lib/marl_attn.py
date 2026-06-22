"""Agent-attention actor-critic (MAT / UPDeT-style) — the 'transformer thing'.

Each agent encodes its local obs with the shared CNN, then a stack of transformer
blocks runs MULTI-HEAD SELF-ATTENTION OVER THE AGENTS (every agent attends to every
other agent's feature), and a head emits per-agent action logits. This is the
permutation-equivariant, variable-N attention core of MAT/UPDeT — NOT a language
model: no tokens, no pretraining, just attention over the team.

Size knobs are module globals (DIM, DEPTH, HEADS) so the runner can sweep them:
  attn-s ~ DIM 32 / DEPTH 1   ·   attn ~ 64/2   ·   attn-l ~ 128/3
Critic is centralized (CTDE), emitting a normalized team value (matches FrontierAttnAC).
"""
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))
import _marl_core as core
import comm_coverage as cc          # noqa: F401  (kept for the __main__ entry / parity)

DIM = 64
DEPTH = 2
HEADS = 4


class Block(eqx.Module):
    """Pre-LN transformer block; self-attention is over the N agent tokens."""
    ln1: eqx.nn.LayerNorm
    qkv: eqx.nn.Linear
    proj: eqx.nn.Linear
    ln2: eqx.nn.LayerNorm
    ff1: eqx.nn.Linear
    ff2: eqx.nn.Linear
    heads: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    def __init__(self, dim, heads, key):
        k = jax.random.split(key, 4)
        self.ln1 = eqx.nn.LayerNorm(dim); self.ln2 = eqx.nn.LayerNorm(dim)
        self.qkv = eqx.nn.Linear(dim, 3 * dim, key=k[0])
        self.proj = eqx.nn.Linear(dim, dim, key=k[1])
        self.ff1 = eqx.nn.Linear(dim, 2 * dim, key=k[2])
        self.ff2 = eqx.nn.Linear(2 * dim, dim, key=k[3])
        self.heads = heads; self.dim = dim

    def __call__(self, x):                                  # x: (N, dim)
        n = x.shape[0]; nh = self.heads; hd = self.dim // nh
        h = jax.vmap(self.ln1)(x)
        qkv = jax.vmap(self.qkv)(h).reshape(n, 3, nh, hd)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]           # (n, nh, hd)
        scores = jnp.einsum("ihd,jhd->hij", q, k) / jnp.sqrt(hd)   # (nh, n, n)
        w = jax.nn.softmax(scores, -1)
        ctx = jnp.einsum("hij,jhd->ihd", w, v).reshape(n, self.dim)
        x = x + jax.vmap(self.proj)(ctx)                    # residual
        h = jax.vmap(self.ln2)(x)
        x = x + jax.vmap(lambda z: self.ff2(jax.nn.relu(self.ff1(z))))(h)
        return x


class AgentAttnAC(eqx.Module):
    enc: core.Encoder            # per-agent CNN → 128
    proj_in: eqx.nn.Linear       # 128 → DIM
    blocks: list                 # DEPTH transformer blocks over the agents
    head: eqx.nn.Linear          # DIM → N_ACTIONS
    critic_enc: core.Encoder     # centralized critic over the global state
    critic_head: eqx.nn.Linear

    def __init__(self, env, key):
        ks = jax.random.split(key, 5 + DEPTH)
        self.enc = core.Encoder(env.obs_channels, ks[0])
        self.proj_in = eqx.nn.Linear(128, DIM, key=ks[1])
        self.head = eqx.nn.Linear(DIM, core.N_ACTIONS, key=ks[2])
        self.critic_enc = core.Encoder(env.global_channels, ks[3])
        self.critic_head = eqx.nn.Linear(128, 1, key=ks[4])
        self.blocks = [Block(DIM, HEADS, ks[5 + i]) for i in range(DEPTH)]

    def logits(self, obs):                                  # (N,C,H,W) → (N,A)
        f = jax.vmap(lambda o: self.proj_in(self.enc(o)))(obs)     # (N, DIM)
        for blk in self.blocks:
            f = blk(f)                                      # self-attention over agents
        return jax.vmap(self.head)(f)

    def values(self, obs, gstate):                          # centralized team value, broadcast
        v = self.critic_head(self.critic_enc(gstate))[0]
        return jnp.full((obs.shape[0],), v)


if __name__ == "__main__":
    core.run(cc.make_env(), AgentAttnAC, "agent-attention (MAT-style)")
