"""
FrontierAttnAC — a frontier-seeking decentralized actor with a CENTRALIZED critic
(CTDE-compatible), the Spec-1 architecture for pushing coverage toward 95%.

Idea: give each agent a *global* "where is unexplored ground" signal via learned
spatial cross-attention, instead of only the fixed 3×3 local-frontier channel.

  conv encoder (size-agnostic) ─► (D, h, w) cell tokens
       tokens tagged with [feature ++ normalized (y,x) ++ uncertainty]
  agent query  = [global-mean-pool(features) ++ agent (y,x)]
  cross-attention(query → cell tokens)  ─► "where to go" context
  action head  = MLP([pooled feature ++ context]) ─► 5 logits

The actor has NO flatten and uses normalized coordinates + global pooling, so it
runs on ANY grid size (the generalization payoff — trained on 16×16, probed on
20×20). The critic stays the fixed-size CTDE critic (training-only, so grid size
doesn't matter for it). Attention is hand-rolled from plain Linears + jax.nn.*
so it threads through lax.scan (eqx.nn.MultiheadAttention / MLP store callables
that the scan carry can't hold — same trap as the joint variant).

Critic outputs a NORMALIZED value (see _frontier_core's return normalization).
"""

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))
import _marl_core as core

D = 64       # token / attention embedding dim
HEADS = 4    # attention heads


class FrontierAttnAC(eqx.Module):
    # ---- actor: size-agnostic conv → cell tokens → frontier cross-attention ----
    c1: eqx.nn.Conv2d
    c2: eqx.nn.Conv2d
    q: eqx.nn.Linear         # query  from [pooled feature ++ agent (y,x)]      (D+2 → D)
    k: eqx.nn.Linear         # key    from [token feature ++ (y,x) ++ uncert]   (D+3 → D)
    v: eqx.nn.Linear         # value  from the same token features              (D+3 → D)
    h1: eqx.nn.Linear        # action head: [pooled ++ context] (2D → D)
    h2: eqx.nn.Linear        #              D → n_actions
    # ---- critic: centralized (CTDE), emits a NORMALIZED team value ----
    critic_enc: core.Encoder
    critic_head: eqx.nn.Linear
    heads: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    def __init__(self, env, key):
        ks = jax.random.split(key, 9)
        self.c1 = eqx.nn.Conv2d(env.obs_channels, 32, 3, stride=1, padding=1, key=ks[0])
        self.c2 = eqx.nn.Conv2d(32, D, 3, stride=2, padding=1, key=ks[1])   # downsample → tokens
        self.q = eqx.nn.Linear(D + 2, D, key=ks[2])
        self.k = eqx.nn.Linear(D + 3, D, key=ks[3])
        self.v = eqx.nn.Linear(D + 3, D, key=ks[4])
        self.h1 = eqx.nn.Linear(D + D, D, key=ks[5])
        self.h2 = eqx.nn.Linear(D, core.N_ACTIONS, key=ks[6])
        self.critic_enc = core.Encoder(env.global_channels, ks[7])
        self.critic_head = eqx.nn.Linear(128, 1, key=ks[8])
        self.heads = HEADS
        self.dim = D

    def _actor_one(self, obs):
        """One agent: (C, H, W) obs → (A,) logits."""
        C, H, W = obs.shape
        x = jax.nn.relu(self.c1(obs))                       # (32, H, W)
        f = jax.nn.relu(self.c2(x))                         # (D, h, w)
        Dc, h, w = f.shape

        # cell tokens tagged with normalized coords + downsampled uncertainty
        tokens = f.reshape(Dc, h * w).T                     # (T, D)
        ys = jnp.linspace(0.0, 1.0, h)[:, None] * jnp.ones((1, w))
        xs = jnp.linspace(0.0, 1.0, w)[None, :] * jnp.ones((h, 1))
        coords = jnp.stack([ys.reshape(-1), xs.reshape(-1)], -1)   # (T, 2)
        unexplored = 1.0 - obs[0]                            # known channel → uncertainty
        unc = jax.image.resize(unexplored, (h, w), method="linear").reshape(-1, 1)  # (T,1)
        tok = jnp.concatenate([tokens, coords, unc], -1)    # (T, D+3)

        # agent query: global summary + agent's own normalized position
        own = obs[1]                                        # one-hot own-position channel
        tot = jnp.maximum(own.sum(), 1e-6)
        ay = (own.sum(1) * jnp.arange(H)).sum() / tot / jnp.maximum(H - 1, 1)
        ax = (own.sum(0) * jnp.arange(W)).sum() / tot / jnp.maximum(W - 1, 1)
        pooled = f.mean((1, 2))                             # (D,)
        qin = jnp.concatenate([pooled, jnp.stack([ay, ax])])   # (D+2,)

        # multi-head cross-attention (query attends over the cell tokens)
        Q = self.q(qin)                                     # (D,)
        K = jax.vmap(self.k)(tok)                           # (T, D)
        V = jax.vmap(self.v)(tok)                           # (T, D)
        nh, hd = self.heads, self.dim // self.heads
        Qh = Q.reshape(nh, hd)
        Kh = K.reshape(-1, nh, hd)
        Vh = V.reshape(-1, nh, hd)
        scores = jnp.einsum("hd,thd->ht", Qh, Kh) / jnp.sqrt(hd)   # (nh, T)
        wts = jax.nn.softmax(scores, -1)
        ctx = jnp.einsum("ht,thd->hd", wts, Vh).reshape(-1)        # (D,)

        return self.h2(jax.nn.relu(self.h1(jnp.concatenate([pooled, ctx]))))   # (A,)

    def logits(self, obs):                  # (N,C,H,W) -> (N,A)
        return jax.vmap(self._actor_one)(obs)

    def values(self, obs, gstate):          # centralized; NORMALIZED team value, broadcast to N
        v = self.critic_head(self.critic_enc(gstate))[0]
        return jnp.full((obs.shape[0],), v)


if __name__ == "__main__":
    import comm_coverage as cc
    import jax
    env = cc.make_env()
    m = FrontierAttnAC(env, jax.random.PRNGKey(0))
    _, st = env.reset(jax.random.PRNGKey(1))
    print("logits", m.logits(env._obs(st)).shape, "values", m.values(env._obs(st), env.global_state(st)).shape)
