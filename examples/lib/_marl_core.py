"""
Shared PPO core for the three centralization variants (independent / CTDE /
joint). The variants differ ONLY in the ActorCritic they pass to ``run`` —
specifically what the actor and critic condition on. Everything here (rollout
collection, GAE, the clipped PPO update, the constraint metrics) is identical
across all three, so the comparison isolates the centralization axis.

Each ActorCritic must expose:
    .logits(obs)        (N,C,H,W) -> (N,A)        per-agent action logits
    .values(obs, g)     (N,C,H,W),(Cg,H,W) -> (N,) state values
"""

from __future__ import annotations

import datetime
import json
import math
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

# ---- hyperparameters -------------------------------------------------------
ENVS, ITERS = 16, 160                  # 16 parallel envs → larger, lower-variance batch
GAMMA, LAM, CLIP, EPOCHS = 0.99, 0.95, 0.2, 4
MINIBATCHES = 4                        # shuffle the batch into 4 minibatches per epoch
LR, ENT_COEF, VF_COEF = 3e-4, 0.01, 0.5
N_ACTIONS = zymera.N_ACTIONS
T = cc.MAX_STEPS

# ---- run artifacts: experiments/marl/<timestamp>/<variant>/ ----------------
EXP_ROOT = Path(__file__).resolve().parents[2] / "experiments" / "marl"


def new_run_dir():
    """Create and return a fresh timestamped run directory under experiments/marl/.
    Pass it to run(..., outdir=dir) so all variants of one invocation share it."""
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    d = EXP_ROOT / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slug(label):
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in label.lower()]
    return "".join(keep).strip("_") or "variant"


def run_config(env, label, seed):
    """Everything needed to reproduce a run (deterministic given seed + code)."""
    return {
        "label": label,
        "seed": seed,
        "algo": {
            "envs": ENVS, "iters": ITERS, "gamma": GAMMA, "lam": LAM,
            "clip": CLIP, "epochs": EPOCHS, "minibatches": MINIBATCHES,
            "lr": LR, "ent_coef": ENT_COEF, "vf_coef": VF_COEF,
            "max_grad_norm": 0.5, "lr_anneal": True, "value_clip": True,
            "adv_norm": True, "batch": ENVS * T, "minibatch": (ENVS * T) // MINIBATCHES,
        },
        "env": {
            "grid_h": int(env.H), "grid_w": int(env.W), "n_agents": int(env.n_agents),
            "max_steps": T, "vis_r": int(env.vis_r), "comm_r": int(env.comm_r),
            "n_obstacles": int(env.n_obstacles), "sense_r": int(env.sense_r),
            "spawn_radius": (None if env.spawn_radius is None else int(env.spawn_radius)),
            "obs_channels": int(env.obs_channels), "global_channels": int(env.global_channels),
        },
        "reward": {"w_cov": float(env.w_cov), "w_conn": float(env.w_conn),
                   "w_overlap": float(env.w_overlap), "w_coll": float(env.w_coll)},
    }


class Encoder(eqx.Module):
    """CNN over a (C, 16, 16) grid → 128-d feature. First conv is stride-1 so
    fine local structure (the local-frontier channel) survives downsampling."""
    c1: eqx.nn.Conv2d
    c2: eqx.nn.Conv2d
    c3: eqx.nn.Conv2d
    lin: eqx.nn.Linear

    def __init__(self, in_ch, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.c1 = eqx.nn.Conv2d(in_ch, 16, 3, stride=1, padding=1, key=k1)   # 16×16
        self.c2 = eqx.nn.Conv2d(16, 32, 3, stride=2, padding=1, key=k2)      # 8×8
        self.c3 = eqx.nn.Conv2d(32, 32, 3, stride=2, padding=1, key=k3)      # 4×4
        self.lin = eqx.nn.Linear(32 * 4 * 4, 128, key=k4)

    def __call__(self, x):
        x = jax.nn.relu(self.c1(x))
        x = jax.nn.relu(self.c2(x))
        x = jax.nn.relu(self.c3(x))
        x = jax.image.resize(x, (x.shape[0], 4, 4), method="linear")  # size-agnostic flatten (any grid → 4×4)
        return jax.nn.relu(self.lin(x.reshape(-1)))


# ---- PPO ------------------------------------------------------------------


def gae(rewards, values, gamma, lam):
    def body(adv_next, t):
        delta = rewards[:, t] + gamma * values[:, t + 1] - values[:, t]
        adv = delta + gamma * lam * adv_next
        return adv, adv
    _, advs = jax.lax.scan(body, jnp.zeros_like(rewards[:, 0]),
                           jnp.arange(T - 1, -1, -1))
    return jnp.moveaxis(advs[::-1], 0, 1)


def _gather_logp(logits, actions):
    logp_all = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(logp_all, actions[..., None], axis=-1)[..., 0]


def ppo_loss(model, obs, gstate, actions, old_logp, adv, ret, old_values):
    # one minibatch: leaves are (M, N, ...); advantages are pre-normalised.
    logits = jax.vmap(model.logits)(obs)               # (M, N, A)
    logp = _gather_logp(logits, actions)               # (M, N)
    entropy = -(jax.nn.softmax(logits, -1) * jax.nn.log_softmax(logits, -1)).sum(-1).mean()
    ratio = jnp.exp(logp - old_logp)
    pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * adv).mean()
    # clipped value loss (PPO): don't let the critic move more than CLIP from its old estimate
    values = jax.vmap(model.values)(obs, gstate)       # (M, N)
    v_clip = old_values + jnp.clip(values - old_values, -CLIP, CLIP)
    vf = jnp.mean(jnp.maximum((values - ret) ** 2, (v_clip - ret) ** 2))
    return pg + VF_COEF * vf - ENT_COEF * entropy


def make_train(env, model):
    # LR annealing: decay linearly to 0 over all optimizer steps (iters × epochs × minibatches)
    lr_schedule = optax.linear_schedule(LR, 0.0, ITERS * EPOCHS * MINIBATCHES)
    opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr_schedule))

    def policy_of(model):
        def policy(obs, key):
            return jax.random.categorical(key, model.logits(obs), axis=-1).astype(jnp.int32)
        return policy

    @eqx.filter_jit
    def train_iter(model, opt_state, key):
        ckey, skey = jax.random.split(key)
        keys = jax.random.split(ckey, ENVS)
        traj = jax.vmap(lambda k: zymera.rollout(env, policy_of(model), T, k))(keys)
        obs = traj["obs"]
        actions = traj["action"]
        rewards = traj["reward"]
        states = traj["world"]
        gstate = jax.vmap(jax.vmap(env.global_state))(states)

        values_all = jax.vmap(jax.vmap(model.values))(obs, gstate)
        old_logp = _gather_logp(jax.vmap(jax.vmap(model.logits))(obs[:, :T]), actions)
        adv = gae(rewards, values_all, GAMMA, LAM)
        old_values = values_all[:, :T]
        ret = adv + old_values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)        # normalise once over the full batch

        # flatten over (envs, time) → B samples; PPO over shuffled minibatches
        B = ENVS * T
        flat = lambda x: x.reshape((B,) + x.shape[2:])
        batch = (flat(obs[:, :T]), flat(gstate[:, :T]), flat(actions),
                 flat(old_logp), flat(adv), flat(ret), flat(old_values))
        mb = B // MINIBATCHES

        def mb_step(carry, idx):
            model, opt_state = carry
            sub = tuple(x[idx] for x in batch)
            loss, grads = eqx.filter_value_and_grad(ppo_loss)(model, *sub)
            updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
            return (eqx.apply_updates(model, updates), opt_state), loss

        def epoch(carry, ekey):
            perm = jax.random.permutation(ekey, B).reshape(MINIBATCHES, mb)
            return jax.lax.scan(mb_step, carry, perm)
        (model, opt_state), losses = jax.lax.scan(
            epoch, (model, opt_state), jax.random.split(skey, EPOCHS))

        coverage = states.team_explored[:, -1].reshape(ENVS, -1).sum(-1) / (cc.GRID ** 2)
        pos = states.pos
        d = jnp.max(jnp.abs(pos[..., :, None, :] - pos[..., None, :, :]), -1)
        off = ~jnp.eye(cc.N_AGENTS, dtype=bool)
        maxd = jnp.where(off, d, 0).max((-1, -2))                      # chain stretch (max pairwise)
        conn = jax.vmap(jax.vmap(lambda dd: cc._connected(dd <= cc.COMM_R)))(d)
        collisions = (((d == 0) & off).sum((-1, -2)) / 2.0)           # collision pairs / step
        redun = (states.explored_by[:, -1].sum((-3, -2, -1))
                 / jnp.maximum(states.team_explored[:, -1].sum((-1, -2)), 1))
        m = (coverage.mean(), conn.mean(), maxd.mean(), redun.mean(), collisions.mean())
        return model, opt_state, jnp.mean(losses), m

    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    return opt_state, train_iter


def early_stop(best, it, last_loss, target_cov, patience, min_iters):
    """Conservative early-exit, returns (stop, reason). Each threshold = None
    disables that check. Deliberately cautious: a run that looks dead can still
    recover (e.g. the w_conn=5 run sat at ~50% for 50 iters then jumped to 90%),
    so the plateau check only fires after a warmup AND a long no-improvement gap.
      (a) NaN/Inf loss        → diverged, bail immediately
      (b) best ≥ target_cov   → we've won, no reason to keep paying
      (c) no new best for `patience` iters, past `min_iters` warmup → plateau"""
    if last_loss is not None and not math.isfinite(last_loss):
        return True, "diverged (non-finite loss)"
    if target_cov is not None and best["cov"] >= target_cov:
        return True, f"hit target {target_cov * 100:.0f}% coverage"
    if (patience is not None and it >= (min_iters or 0)
            and (it - best["iter"]) >= patience):
        return True, f"plateau — no best in {patience} iters (best @ {best['iter']})"
    return False, ""


def run(env, build_model, label, seed=1, outdir=None,
        target_cov=0.985, patience=70, min_iters=110):
    """Train one variant; print progress; persist artifacts; return the
    best-coverage record (augmented with the output directory).

    Early-exit (``early_stop``): stops when coverage hits ``target_cov``, the
    loss diverges, or best coverage plateaus for ``patience`` iters past the
    ``min_iters`` warmup. Set any to None to disable.

    Writes ``<outdir>/<variant>/``:
      * ``metrics.npz``   — full per-iteration arrays (loss + all 5 metrics)
      * ``checkpoint.eqx``— the model that achieved best coverage (reloadable)
      * ``config.json``   — hyperparameters + env + reward weights + seed
      * ``best.json``     — the single best-coverage record
    If ``outdir`` is None a fresh experiments/marl/<timestamp>/ dir is made."""
    model = build_model(env, jax.random.PRNGKey(0))
    opt_state, train_iter = make_train(env, model)
    print(f"{label} · {ENVS} envs · {T} steps · {ITERS} iters · "
          f"{cc.N_AGENTS} agents on {cc.GRID}×{cc.GRID}")
    key = jax.random.PRNGKey(seed)
    best = {"cov": -1.0, "iter": 0}
    best_model = model
    hist = {k: [] for k in ("loss", "cov", "conn", "maxd", "redun", "coll")}
    for it in range(ITERS):
        key, sk = jax.random.split(key)
        pre = model                          # the model that generates this iter's metrics
        model, opt_state, loss, m = train_iter(model, opt_state, sk)
        cov, conn, maxd, redun, coll = (float(x) for x in m)
        for k, v in zip(("loss", "cov", "conn", "maxd", "redun", "coll"),
                        (float(loss), cov, conn, maxd, redun, coll)):
            hist[k].append(v)
        if cov > best["cov"]:
            best = dict(cov=cov, conn=conn, maxd=maxd, redun=redun, coll=coll, iter=it)
            best_model = pre                 # checkpoint the model that ACHIEVED best cov
        stop, reason = early_stop(best, it, float(loss), target_cov, patience, min_iters)
        if it % 10 == 0 or it == ITERS - 1 or stop:
            print(f"  iter {it:3d}  cov {cov*100:5.1f}%  connected {conn*100:4.0f}%  "
                  f"max-dist {maxd:4.1f}  redundancy {redun:.2f}×  collisions {coll:.2f}")
        if stop:
            print(f"  ⏹ early exit at iter {it} (of {ITERS}): {reason}")
            break
    print(f"  ► BEST {label}: cov {best['cov']*100:.1f}%  connected {best['conn']*100:.0f}%"
          f"  max-dist {best['maxd']:.1f} (chain if >{cc.COMM_R})  redundancy {best['redun']:.2f}×"
          f"  collisions {best['coll']:.2f}  (iter {best['iter']})")

    # ---- persist -----------------------------------------------------------
    base = outdir if outdir is not None else new_run_dir()
    vdir = Path(base) / _slug(label)
    vdir.mkdir(parents=True, exist_ok=True)
    np.savez(vdir / "metrics.npz", iter=np.arange(len(hist["loss"]), dtype=np.int32),
             **{k: np.asarray(v, dtype=np.float32) for k, v in hist.items()})
    eqx.tree_serialise_leaves(str(vdir / "checkpoint.eqx"), best_model)
    (vdir / "config.json").write_text(json.dumps(run_config(env, label, seed), indent=2))
    (vdir / "best.json").write_text(json.dumps(best, indent=2))
    print(f"    saved → {vdir}  (metrics.npz · checkpoint.eqx · config.json · best.json)")
    best["dir"] = str(vdir)
    return best
