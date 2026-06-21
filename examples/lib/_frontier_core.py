"""
PPO training core for the FrontierAttn experiments, WITH return normalization
(ValueNorm) AND optional HARD-collision via ACTION MASKING.

Action masking (mask_collisions=True): the policy never *proposes* a colliding
move — sequential masked sampling claims cells in agent order and masks out any
action whose target cell is already claimed (STAY is always valid, so no
deadlock). This guarantees no two agents share a cell while keeping PPO's
credit assignment intact (the masked log-probs are used in both rollout and
loss). This replaces the env's silent action-override (`hard_collision`), which
corrupts training when collisions are frequent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).parent))
import comm_coverage as cc
import zymera
import _marl_core as core
from _marl_core import (ENVS, GAMMA, LAM, CLIP, EPOCHS, MINIBATCHES, LR,
                        ENT_COEF, VF_COEF, T, gae, _gather_logp, new_run_dir,
                        _slug, run_config, early_stop)

VN_BETA = 0.95   # running-stats momentum for the return normalizer


# ---- return normalizer (debiased running mean/var of returns) --------------
def vn_init():
    return (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))


def _vn_stats(vn):
    rm, rmsq, db = vn
    db = jnp.maximum(db, 1e-8)
    mean = rm / db
    var = jnp.maximum(rmsq / db - mean ** 2, 1e-2)
    return mean, jnp.sqrt(var)


def vn_update(vn, x):
    rm, rmsq, db = vn
    return (rm * VN_BETA + x.mean() * (1 - VN_BETA),
            rmsq * VN_BETA + (x ** 2).mean() * (1 - VN_BETA),
            db * VN_BETA + (1 - VN_BETA))


def vn_norm(x, vn):
    mean, std = _vn_stats(vn)
    return (x - mean) / std


def vn_denorm(x, vn):
    mean, std = _vn_stats(vn)
    return x * std + mean


# ---- HARD collision via action masking -------------------------------------
def _targets(pos, wall, H, W):
    """(N, A, 2) wall/bound-clipped target cell per agent×action — matches env._move."""
    t = pos[:, None, :] + cc._DELTAS[None, :, :]
    r = jnp.clip(t[..., 0], 0, H - 1)
    c = jnp.clip(t[..., 1], 0, W - 1)
    hit = wall[r, c]
    r = jnp.where(hit, pos[:, None, 0], r)
    c = jnp.where(hit, pos[:, None, 1], c)
    return jnp.stack([r, c], -1)


def _conn_ok(positions, comm_r):
    """(scalar bool) is the comm graph (Chebyshev ≤ comm_r) a single connected component?"""
    d = jnp.max(jnp.abs(positions[:, None, :] - positions[None, :, :]), -1)
    return cc._connected(d <= comm_r)


def _move_mask(committed, ti, i, comm_r):
    """(A,) bool valid-move mask: no collision with a committed cell, and (if comm_r set)
    the move keeps the comm graph connected. STAY (action 0) is always kept as a fallback."""
    A, N = ti.shape[0], committed.shape[0]
    eq = jnp.all(ti[:, None, :] == committed[None, :, :], -1)     # (A,N)
    valid = ~(eq.at[:, i].set(False)).any(-1)                     # collision-valid
    if comm_r is not None:                                         # HARD connectivity guardrail
        cand = jnp.broadcast_to(committed, (A, N, 2)).at[:, i, :].set(ti)   # (A,N,2)
        conn = jax.vmap(lambda p: _conn_ok(p, comm_r))(cand)      # (A,)
        conn = conn.at[0].set(True)                               # STAY always allowed
        cv = valid & conn
        valid = jnp.where(cv.any(), cv, valid)                    # never deadlock
    return valid


def _masked_sample(logits, pos, wall, key, H, W, comm_r=None):
    """Sequential masked sampling → (actions, masked logp). Masks collisions, and — when
    comm_r is set — any move that would disconnect the comm graph (hard integrity guardrail)."""
    N, A = logits.shape
    tg = _targets(pos, wall, H, W)

    def body(carry, i):
        committed, key = carry
        ti = tg[i]                                                # (A, 2)
        valid = _move_mask(committed, ti, i, comm_r)              # (A,)
        ml = jnp.where(valid, logits[i], -1e9)
        key, sk = jax.random.split(key)
        a = jax.random.categorical(sk, ml)
        return (committed.at[i].set(ti[a]), key), (a, jax.nn.log_softmax(ml)[a])

    (_, _), (actions, logps) = jax.lax.scan(body, (pos, key), jnp.arange(N))
    return actions.astype(jnp.int32), logps


def _masked_logp_ent(logits, pos, wall, actions, H, W, comm_r=None):
    """Recompute masked logp (N,) + entropy (scalar) for stored actions — same mask
    (collision + optional connectivity guardrail) as _masked_sample."""
    N, A = logits.shape
    tg = _targets(pos, wall, H, W)

    def body(committed, i):
        ti = tg[i]
        valid = _move_mask(committed, ti, i, comm_r)
        ml = jnp.where(valid, logits[i], -1e9)
        lp = jax.nn.log_softmax(ml)
        ent = -(jnp.where(valid, jnp.exp(lp) * lp, 0.0)).sum()
        return committed.at[i].set(ti[actions[i]]), (lp[actions[i]], ent)

    _, (logps, ents) = jax.lax.scan(body, pos, jnp.arange(N))
    return logps, ents.mean()


def _rollout_masked(env, model, T, key, comm_r=None):
    """Rollout sampling collision-free actions via masking. Same dict shape as
    zymera.rollout (obs/world T+1, action/reward/logp T) plus masked log-probs.
    comm_r set ⇒ also apply the hard connectivity guardrail."""
    rk, key = jax.random.split(key)
    _, state0 = env.reset(rk)
    H, W = env.H, env.W

    def step_fn(carry, _):
        state, key = carry
        key, ak, sk = jax.random.split(key, 3)
        obs = env._obs(state)
        actions, logp = _masked_sample(model.logits(obs), state.pos, state.wall, ak, H, W, comm_r)
        _, nstate, reward, _, _ = env.step(state, actions, sk)
        return (nstate, key), (obs, actions, logp, reward, state)

    (final_state, _), (obs_s, act_s, logp_s, rew_s, world_s) = jax.lax.scan(
        step_fn, (state0, key), None, length=T)
    obs_all = jnp.concatenate([obs_s, env._obs(final_state)[None]], 0)
    world_all = jax.tree_util.tree_map(
        lambda a, b: jnp.concatenate([a, b[None]], 0), world_s, final_state)
    return dict(obs=obs_all, action=act_s, logp=logp_s, reward=rew_s, world=world_all)


# ---- PPO loss (value term in NORMALIZED space) -----------------------------
def ppo_loss(model, obs, gstate, actions, old_logp, adv, ret_norm, old_v_norm):
    logits = jax.vmap(model.logits)(obs)
    logp = _gather_logp(logits, actions)
    entropy = -(jax.nn.softmax(logits, -1) * jax.nn.log_softmax(logits, -1)).sum(-1).mean()
    ratio = jnp.exp(logp - old_logp)
    pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * adv).mean()
    v = jax.vmap(model.values)(obs, gstate)
    v_clip = old_v_norm + jnp.clip(v - old_v_norm, -CLIP, CLIP)
    vf = jnp.mean(jnp.maximum((v - ret_norm) ** 2, (v_clip - ret_norm) ** 2))
    return pg + VF_COEF * vf - ENT_COEF * entropy


def make_train(env, model, mask_collisions=False, guard_conn=False):
    lr_schedule = optax.linear_schedule(LR, 0.0, core.ITERS * EPOCHS * MINIBATCHES)
    opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr_schedule))
    H, W = env.H, env.W
    guard_r = int(env.comm_r) if guard_conn else None   # hard connectivity guardrail (None=off)
    use_mask = mask_collisions or guard_conn            # both run through the sequential masker

    def policy_of(model):
        def policy(obs, key):
            return jax.random.categorical(key, model.logits(obs), axis=-1).astype(jnp.int32)
        return policy

    def ppo_loss_masked(model, obs, gstate, actions, old_logp, adv, ret_norm, old_v_norm, pos, wall):
        logits = jax.vmap(model.logits)(obs)                  # (M, N, A)
        logp, ent = jax.vmap(lambda lg, p, w, a: _masked_logp_ent(lg, p, w, a, H, W, guard_r))(
            logits, pos, wall, actions)                       # (M,N), (M,)
        ratio = jnp.exp(logp - old_logp)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * adv).mean()
        v = jax.vmap(model.values)(obs, gstate)
        v_clip = old_v_norm + jnp.clip(v - old_v_norm, -CLIP, CLIP)
        vf = jnp.mean(jnp.maximum((v - ret_norm) ** 2, (v_clip - ret_norm) ** 2))
        return pg + VF_COEF * vf - ENT_COEF * ent.mean()

    @eqx.filter_jit
    def train_iter(model, opt_state, vn, key):
        ckey, skey = jax.random.split(key)
        keys = jax.random.split(ckey, ENVS)
        if use_mask:
            traj = jax.vmap(lambda k: _rollout_masked(env, model, T, k, guard_r))(keys)
        else:
            traj = jax.vmap(lambda k: zymera.rollout(env, policy_of(model), T, k))(keys)
        obs, actions, rewards, states = traj["obs"], traj["action"], traj["reward"], traj["world"]
        gstate = jax.vmap(jax.vmap(env.global_state))(states)

        v_norm_all = jax.vmap(jax.vmap(model.values))(obs, gstate)
        v_real_all = vn_denorm(v_norm_all, vn)
        if use_mask:
            old_logp = traj["logp"]                                   # masked, from rollout
        else:
            old_logp = _gather_logp(jax.vmap(jax.vmap(model.logits))(obs[:, :T]), actions)
        adv = gae(rewards, v_real_all, GAMMA, LAM)
        ret_real = adv + v_real_all[:, :T]
        vn_new = vn_update(vn, ret_real)
        ret_norm = vn_norm(ret_real, vn_new)
        old_v_norm = v_norm_all[:, :T]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        B = ENVS * T
        flat = lambda x: x.reshape((B,) + x.shape[2:])
        common = [flat(obs[:, :T]), flat(gstate[:, :T]), flat(actions),
                  flat(old_logp), flat(adv), flat(ret_norm), flat(old_v_norm)]
        if use_mask:
            batch = tuple(common + [flat(states.pos[:, :T]), flat(states.wall[:, :T])])
            loss_fn = ppo_loss_masked
        else:
            batch = tuple(common)
            loss_fn = ppo_loss
        mb = B // MINIBATCHES

        def mb_step(carry, idx):
            model, opt_state = carry
            sub = tuple(x[idx] for x in batch)
            loss, grads = eqx.filter_value_and_grad(loss_fn)(model, *sub)
            updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
            return (eqx.apply_updates(model, updates), opt_state), loss

        def epoch(carry, ekey):
            perm = jax.random.permutation(ekey, B).reshape(MINIBATCHES, mb)
            return jax.lax.scan(mb_step, carry, perm)
        (model, opt_state), losses = jax.lax.scan(
            epoch, (model, opt_state), jax.random.split(skey, EPOCHS))

        _flat = states.team_explored[:, -1].reshape(ENVS, -1)
        coverage = _flat.sum(-1) / _flat.shape[-1]      # divide by ACTUAL cell count (H*W), any world size
        pos = states.pos
        d = jnp.max(jnp.abs(pos[..., :, None, :] - pos[..., None, :, :]), -1)
        off = ~jnp.eye(cc.N_AGENTS, dtype=bool)
        maxd = jnp.where(off, d, 0).max((-1, -2))
        conn = jax.vmap(jax.vmap(lambda dd: cc._connected(dd <= cc.COMM_R)))(d)
        collisions = (((d == 0) & off).sum((-1, -2)) / 2.0)
        redun = (states.explored_by[:, -1].sum((-3, -2, -1))
                 / jnp.maximum(states.team_explored[:, -1].sum((-1, -2)), 1))
        m = (coverage.mean(), conn.mean(), maxd.mean(), redun.mean(), collisions.mean())
        return model, opt_state, vn_new, jnp.mean(losses), m

    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    return opt_state, train_iter


def run(env, build_model, label, seed=1, outdir=None,
        target_cov=0.985, patience=70, min_iters=110, mask_collisions=False,
        init_model=None, guard_conn=False):
    """Train with return normalization (+ optional hard-collision action masking and/or
    hard connectivity guardrail); print progress; persist artifacts; return the
    best-coverage record. ``init_model`` warm-starts from an existing policy."""
    model = init_model if init_model is not None else build_model(env, jax.random.PRNGKey(0))
    opt_state, train_iter = make_train(env, model, mask_collisions, guard_conn)
    vn = vn_init()
    print(f"{label} · {ENVS} envs · {T} steps · {core.ITERS} iters · "
          f"{cc.N_AGENTS} agents on {cc.GRID}×{cc.GRID} · return-norm ON"
          f"{' · MASKED hard-collision' if mask_collisions else ''}")
    key = jax.random.PRNGKey(seed)
    best = {"cov": -1.0, "iter": 0}
    best_model = model
    hist = {k: [] for k in ("loss", "cov", "conn", "maxd", "redun", "coll")}
    for it in range(core.ITERS):
        key, sk = jax.random.split(key)
        pre = model
        model, opt_state, vn, loss, m = train_iter(model, opt_state, vn, sk)
        cov, conn, maxd, redun, coll = (float(x) for x in m)
        for k, val in zip(("loss", "cov", "conn", "maxd", "redun", "coll"),
                          (float(loss), cov, conn, maxd, redun, coll)):
            hist[k].append(val)
        if cov > best["cov"]:
            best = dict(cov=cov, conn=conn, maxd=maxd, redun=redun, coll=coll, iter=it)
            best_model = pre
        stop, reason = early_stop(best, it, float(loss), target_cov, patience, min_iters)
        if it % 10 == 0 or it == core.ITERS - 1 or stop:
            print(f"  iter {it:3d}  cov {cov*100:5.1f}%  connected {conn*100:4.0f}%  "
                  f"max-dist {maxd:4.1f}  redundancy {redun:.2f}×  collisions {coll:.2f}")
        if stop:
            print(f"  ⏹ early exit at iter {it} (of {core.ITERS}): {reason}")
            break
    print(f"  ► BEST {label}: cov {best['cov']*100:.1f}%  connected {best['conn']*100:.0f}%"
          f"  max-dist {best['maxd']:.1f} (chain if >{cc.COMM_R})  redundancy {best['redun']:.2f}×"
          f"  collisions {best['coll']:.2f}  (iter {best['iter']})")

    base = outdir if outdir is not None else new_run_dir()
    vdir = Path(base) / _slug(label)
    vdir.mkdir(parents=True, exist_ok=True)
    np.savez(vdir / "metrics.npz", iter=np.arange(len(hist["loss"]), dtype=np.int32),
             **{k: np.asarray(v, dtype=np.float32) for k, v in hist.items()})
    eqx.tree_serialise_leaves(str(vdir / "checkpoint.eqx"), best_model)
    cfg = run_config(env, label, seed)
    cfg["algo"]["return_norm"] = True
    cfg["algo"]["mask_collisions"] = bool(mask_collisions)
    cfg["algo"]["iters"] = core.ITERS
    (vdir / "config.json").write_text(json.dumps(cfg, indent=2))
    (vdir / "best.json").write_text(json.dumps(best, indent=2))
    print(f"    saved → {vdir}")
    best["dir"] = str(vdir)
    return best
