"""
FrontierCommAttnAC — FrontierAttnAC + a SECOND attention head: inter-agent
(comm) attention, where each agent attends over its in-range neighbors.

Purpose: test the hypothesis that attention is *capability, not incentive*. The
agent can now richly perceive/process its neighbors — but if trained with NO
connectivity/collision reward, it should STILL disperse, because nothing makes
keeping min/max distance valuable. (The plain FrontierAttn already had neighbour
positions in its obs and dispersed to 32% connectivity — this makes the neighbour
channel a full learned attention and asks: does that change anything? Prediction: no.)

Two attention heads:
  spatial frontier attention : query → grid cells weighted by uncertainty  ("where to go")
  inter-agent comm attention : query → in-range neighbours' features        ("who's around")
Action head conditions on [own feature ++ frontier context ++ neighbour context].
"""

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))
import _marl_core as core

D = 64
HEADS = 4


class FrontierCommAttnAC(eqx.Module):
    # shared conv encoder
    c1: eqx.nn.Conv2d
    c2: eqx.nn.Conv2d
    # spatial frontier attention
    q: eqx.nn.Linear
    k: eqx.nn.Linear
    v: eqx.nn.Linear
    # inter-agent (comm) attention
    nq: eqx.nn.Linear
    nk: eqx.nn.Linear
    nv: eqx.nn.Linear
    # action head: [pooled (++ frontier_ctx) (++ neighbour_ctx)] → D → A
    h1: eqx.nn.Linear
    h2: eqx.nn.Linear
    # centralized critic (NORMALIZED value, for _frontier_core)
    critic_enc: core.Encoder
    critic_head: eqx.nn.Linear
    per_agent_head: eqx.nn.Linear     # per-agent value head [global ++ pooled] → 1
    heads: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    comm_r: int = eqx.field(static=True)
    # ablation flags (static)
    use_frontier: bool = eqx.field(static=True)
    use_comm: bool = eqx.field(static=True)
    per_agent_critic: bool = eqx.field(static=True)

    def __init__(self, env, key, *, use_frontier=True, use_comm=True,
                 per_agent_critic=False, hi_res=False):
        ks = jax.random.split(key, 13)
        self.c1 = eqx.nn.Conv2d(env.obs_channels, 32, 3, stride=1, padding=1, key=ks[0])
        self.c2 = eqx.nn.Conv2d(32, D, 3, stride=(1 if hi_res else 2), padding=1, key=ks[1])
        self.q = eqx.nn.Linear(D + 2, D, key=ks[2])
        self.k = eqx.nn.Linear(D + 3, D, key=ks[3])
        self.v = eqx.nn.Linear(D + 3, D, key=ks[4])
        self.nq = eqx.nn.Linear(D, D, key=ks[5])
        self.nk = eqx.nn.Linear(D, D, key=ks[6])
        self.nv = eqx.nn.Linear(D, D, key=ks[7])
        h1_in = (1 + int(use_frontier) + int(use_comm)) * D     # which contexts feed the action head
        self.h1 = eqx.nn.Linear(h1_in, D, key=ks[8])
        self.h2 = eqx.nn.Linear(D, core.N_ACTIONS, key=ks[9])
        self.critic_enc = core.Encoder(env.global_channels, ks[10])
        self.critic_head = eqx.nn.Linear(128, 1, key=ks[11])
        self.per_agent_head = eqx.nn.Linear(128 + D, 1, key=ks[12])
        self.heads = HEADS
        self.dim = D
        self.comm_r = env.comm_r
        self.use_frontier, self.use_comm, self.per_agent_critic = use_frontier, use_comm, per_agent_critic

    def _agent_feature(self, obs):
        """One agent: (C,H,W) → (pooled feature D, frontier context D, cell pos (2,))."""
        C, H, W = obs.shape
        x = jax.nn.relu(self.c1(obs))
        f = jax.nn.relu(self.c2(x))                          # (D, h, w)
        Dc, h, w = f.shape
        tokens = f.reshape(Dc, h * w).T
        ys = jnp.linspace(0.0, 1.0, h)[:, None] * jnp.ones((1, w))
        xs = jnp.linspace(0.0, 1.0, w)[None, :] * jnp.ones((h, 1))
        coords = jnp.stack([ys.reshape(-1), xs.reshape(-1)], -1)
        unc = jax.image.resize(1.0 - obs[0], (h, w), method="linear").reshape(-1, 1)
        tok = jnp.concatenate([tokens, coords, unc], -1)     # (T, D+3)
        own = obs[1]
        tot = jnp.maximum(own.sum(), 1e-6)
        ay = (own.sum(1) * jnp.arange(H)).sum() / tot        # CELL coords (for comm mask)
        ax = (own.sum(0) * jnp.arange(W)).sum() / tot
        pooled = f.mean((1, 2))
        qin = jnp.concatenate([pooled, jnp.stack([ay / jnp.maximum(H - 1, 1),
                                                  ax / jnp.maximum(W - 1, 1)])])
        Q = self.q(qin)
        K = jax.vmap(self.k)(tok)
        V = jax.vmap(self.v)(tok)
        nh, hd = self.heads, self.dim // self.heads
        scores = jnp.einsum("hd,thd->ht", Q.reshape(nh, hd), K.reshape(-1, nh, hd)) / jnp.sqrt(hd)
        ctx = jnp.einsum("ht,thd->hd", jax.nn.softmax(scores, -1), V.reshape(-1, nh, hd)).reshape(-1)
        return pooled, ctx, jnp.stack([ay, ax])

    def logits(self, obs):                                   # (N,C,H,W) -> (N,A)
        N = obs.shape[0]
        pooled, fctx, pos = jax.vmap(self._agent_feature)(obs)   # (N,D),(N,D),(N,2)
        parts = [pooled]
        if self.use_frontier:
            parts.append(fctx)
        if self.use_comm:
            # inter-agent attention masked to in-range neighbours (self always included)
            d = jnp.max(jnp.abs(pos[:, None, :] - pos[None, :, :]), -1)
            mask = d <= self.comm_r
            nh, hd = self.heads, self.dim // self.heads
            Q = jax.vmap(self.nq)(pooled).reshape(N, nh, hd)
            K = jax.vmap(self.nk)(pooled).reshape(N, nh, hd)
            V = jax.vmap(self.nv)(pooled).reshape(N, nh, hd)
            scores = jnp.einsum("inh,jnh->nij", Q, K) / jnp.sqrt(hd)   # (nh, N, N)
            scores = jnp.where(mask[None], scores, -1e9)
            wts = jax.nn.softmax(scores, -1)
            nctx = jnp.einsum("nij,jnh->inh", wts, V).reshape(N, self.dim)   # (N, D)
            parts.append(nctx)
        head_in = jnp.concatenate(parts, -1)                    # (N, (1+uf+uc)*D)
        return jax.vmap(lambda z: self.h2(jax.nn.relu(self.h1(z))))(head_in)

    def values(self, obs, gstate):                           # centralized NORMALIZED value
        g = self.critic_enc(gstate)                          # (128,) global feature
        if self.per_agent_critic:                            # per-agent value: [global ++ own pooled]
            pooled, _, _ = jax.vmap(self._agent_feature)(obs)    # (N,D)
            vin = jax.vmap(lambda p: jnp.concatenate([g, p]))(pooled)
            return jax.vmap(self.per_agent_head)(vin)[:, 0]      # (N,)
        v = self.critic_head(g)[0]                           # shared team value
        return jnp.full((obs.shape[0],), v)


if __name__ == "__main__":
    import comm_coverage as cc
    env = cc.make_env()
    m = FrontierCommAttnAC(env, jax.random.PRNGKey(0))
    _, st = env.reset(jax.random.PRNGKey(1))
    print("logits", m.logits(env._obs(st)).shape, "values", m.values(env._obs(st), env.global_state(st)).shape)
