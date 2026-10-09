"""Pretrain the Gaussian-head BNN on the DEFAULT Reacher-v5 dynamics.

Same pipeline as pretrain_ant.py / pretrain_pendulum.py, adapted to Reacher's
10-dim observation + 2-dim action (-> 11 target dims) with four Reacher-specific
changes:

1. IK-TELEPORT COLLECTION.  Reacher's goal region is "fingertip ON the target",
   and random actions essentially never produce it: the arm flails, and the
   fingertip-target distance under random torques averages ~0.13 with only ~2%
   of steps inside 0.03.  A model trained on that data is worst exactly where
   the planner needs to be sharp -- the last centimetre of the reach, where the
   reward gradient lives.  We therefore teleport every TELEPORT_EVERY steps to a
   state built BACKWARDS: sample the target from the env's own distribution
   (uniform in the disk r < 0.2), sample a desired fingertip point near it, then
   solve the two-link inverse kinematics for the joint angles that put the
   fingertip there.  Because the target is drawn from the env's distribution
   first, this leaves the target marginal EXACTLY on-distribution while giving
   us free control over the fingertip-target distance -- which a forward
   "place the target near wherever the arm happens to be" scheme would not.

2. GOAL-WEIGHTED MINIBATCHES (CLAUDE.md: "pretraining should prioritize the
   transitions close to the goal more").  See goal_weights(): exponential in the
   fingertip-target distance.

3. AN OUTPUT SCALER with a floor.  The 11 target dims span
   delta-target (identically 0, std 0) through delta-qvel (std ~ 4), so
   standardizing is essential -- and the two identically-zero target dims need a
   floor or their standardized targets divide by ~0.  See --sd-floor-frac.

4. NO TRUNCATION DURING COLLECTION.  Reacher-v5 ships a 50-step TimeLimit; we
   build the collection env with it disabled and rely on teleports for state
   diversity, so no transition is ever cut short or contaminated by a reset.

Saves checkpoint + normalizer + arch.json to data/reacher/.

Run:  python3 pretrain_reacher.py
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch
from torch import optim

from config import device
from bnn.gaussian_model import make_gaussian_bnn
from env.reacher import (build_reacher_env, fingertip_xy, OBS_DIM, ACT_DIM,
                         DIST_IDX, QVEL_IDX, TARGET_IDX, L0, L1, TARGET_RADIUS)
from mbrl.types import TransitionBatch

_HERE = pathlib.Path(__file__).parent
SAVE_DIR = _HERE / "data" / "reacher"

N_COLLECT = 100_000
N_VAL = 20_000
BATCH_SIZE = 2048
N_EPOCHS = 1200          # 400 was clearly undertrained: val NRMSE 0.137 @400 vs 0.064
                         # @1200, reward-head RMSE 0.064 -> 0.037, and calibration
                         # z_std 0.42 -> 0.55.  Cost is only ~6 min on one GPU.
LR = 1e-3
BETA = 1.0
KL_BUDGET = 7.8           # architecture-independent KL weight (see gaussian_model)

# ── collection ────────────────────────────────────────────────────────────────
TELEPORT_EVERY = 10       # steps between state teleports (episodes are only 50)
P_GOAL = 0.45             # teleports placing the fingertip ~ON the target
P_NEAR = 0.25             # ... in the neighbourhood of the target
ACT_RHO = 0.7             # AR(1) action smoothing; 0 is iid uniform noise
REACH_MIN = L1 - L0 + 1e-3    # 0.011: inner radius of the reachable annulus
REACH_MAX = L0 + L1 - 1e-3    # 0.209: outer radius

# ── goal-weighting ────────────────────────────────────────────────────────────
D0 = 0.05                 # distance scale (m) of the exponential emphasis
FLOOR = 0.05              # min weight so off-goal dynamics still train


def _ik(px, py, rng):
    """Two-link planar inverse kinematics: fingertip (px, py) -> (theta0, theta1).

    cos(theta1) = (r^2 - L0^2 - L1^2) / (2 L0 L1), with the elbow-up / elbow-down
    branch chosen at random so both halves of the configuration space are covered.
    The caller must supply a point inside the reachable annulus; we clamp the
    cosine anyway so numerical edge cases cannot produce a nan.
    """
    r2 = px * px + py * py
    c1 = np.clip((r2 - L0 ** 2 - L1 ** 2) / (2 * L0 * L1), -1.0, 1.0)
    th1 = np.arccos(c1) * (1.0 if rng.random() < 0.5 else -1.0)
    th0 = np.arctan2(py, px) - np.arctan2(L1 * np.sin(th1), L0 + L1 * np.cos(th1))
    return th0, th1


def _sample_target(rng):
    """Draw a goal exactly as ReacherEnv.reset_model does: uniform in [-0.2,0.2]^2
    rejected onto the disk of radius 0.2."""
    while True:
        g = rng.uniform(-0.2, 0.2, size=2)
        if np.linalg.norm(g) < TARGET_RADIUS:
            return g


def _sample_state(env, rng, kind):
    """Draw a (qpos, qvel) for a teleport.  kind: 'goal' | 'near' | 'broad'.

    qpos = [theta0, theta1, target_x, target_y]; qvel = [dtheta0, dtheta1, 0, 0]
    (the target's two slide joints are unactuated and must stay at rest).
    """
    tgt = _sample_target(rng)
    if kind == "broad":
        # Any arm configuration at all, including fast flailing -- the states a
        # random or badly-planned action sequence actually visits.
        th0 = rng.uniform(-np.pi, np.pi)
        th1 = rng.uniform(-3.0, 3.0)
        qvel = rng.normal(0.0, 8.0, size=2)
    else:
        # Place the FINGERTIP at a controlled distance from the on-distribution
        # target, then solve IK for the joints that put it there.
        sd = 0.015 if kind == "goal" else 0.07
        for _ in range(20):
            p = tgt + rng.normal(0.0, sd, size=2)
            r = np.linalg.norm(p)
            if REACH_MIN <= r <= REACH_MAX:
                break
        else:                                   # target hugging the reach limit
            r = np.clip(np.linalg.norm(p), REACH_MIN, REACH_MAX)
            p = p / max(np.linalg.norm(p), 1e-9) * r
        th0, th1 = _ik(p[0], p[1], rng)
        # Slow near the goal (the arm must SETTLE on the target, not fly past it),
        # moderate in the neighbourhood (the approach phase).
        qvel = rng.normal(0.0, 1.5 if kind == "goal" else 5.0, size=2)
    qpos = np.array([th0, th1, tgt[0], tgt[1]], dtype=np.float64)
    return qpos, np.array([qvel[0], qvel[1], 0.0, 0.0], dtype=np.float64)


def collect_data(n_steps, rng, gear=1.0, mass=1.0, damping=1.0,
                 teleport_every=TELEPORT_EVERY):
    """Collect DEFAULT-dynamics transitions with broad + goal-focused coverage.

    max_episode_steps is disabled so the 50-step TimeLimit never truncates a
    trajectory mid-collection; the teleports supply all the state diversity.
    """
    env = build_reacher_env([(0, gear)], [(0, mass)], [(0, damping)],
                            max_episode_steps=10 ** 9)
    obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
    lo, hi = env.action_space.low, env.action_space.high
    act = np.zeros(ACT_DIM, dtype=np.float32)
    data = []
    for i in range(n_steps):
        if i % teleport_every == 0:
            u = rng.random()
            kind = "goal" if u < P_GOAL else ("near" if u < P_GOAL + P_NEAR
                                              else "broad")
            obs = env.set_state_obs(*_sample_state(env, rng, kind))
            act = np.zeros(ACT_DIM, dtype=np.float32)
        # AR(1)-smoothed actions: correlated torques produce sustained pushes
        # that iid uniform noise averages away, while still covering the box.
        eps = rng.uniform(lo, hi).astype(np.float32)
        act = (ACT_RHO * act + np.sqrt(1 - ACT_RHO ** 2) * eps).astype(np.float32)
        act = np.clip(act, lo, hi)
        next_obs, rew, term, trunc, _ = env.step(act)
        if not np.all(np.isfinite(next_obs)):        # MuJoCo blow-up: re-teleport
            obs = env.set_state_obs(*_sample_state(env, rng, "broad"))
            continue
        data.append((obs.copy(), act.copy(), next_obs.copy(), rew))
        obs = next_obs
    return data


def goal_weights(obs_arr, floor=FLOOR, d0=D0):
    """Per-sample sampling weights emphasising states with the fingertip near the
    target -- Reacher's analogue of the pendulum's near-upright weighting and
    Ant's upright/forward weighting.

        w = exp(-d / d0) + floor,        d = ||obs[8:10]||

    A fingertip sitting on the target scores 1.05 against 0.07 for one at the
    far side of the workspace (d = 0.2) -- a ~15x emphasis, matching the ratio
    the pendulum and Ant scripts use -- while the floor keeps every sample
    reachable so the planner's imagined excursions are still modelled.

    Distance alone defines the goal here: the reward is -d - ||a||^2 with no
    velocity term, so weighting by speed as well would only starve the approach
    phase, which is most of what the planner has to get right.
    """
    d = np.linalg.norm(obs_arr[:, DIST_IDX], axis=1)
    w = np.exp(-d / d0) + floor
    return (w / w.sum()).astype(np.float64)


def to_arrays(data):
    obs = np.stack([d[0] for d in data]).astype(np.float32)
    act = np.stack([d[1] for d in data]).astype(np.float32)
    nxt = np.stack([d[2] for d in data]).astype(np.float32)
    rew = np.array([d[3] for d in data], dtype=np.float32)
    return obs, act, nxt, rew


@torch.no_grad()
def eval_split(bnn, xs, ys):
    """Deterministic (mean-weight) held-out metrics in RAW target units."""
    mean, logvar = bnn._run_network(xs, sample=False)
    err = ys - mean
    nll = 0.5 * (logvar + err ** 2 / logvar.exp()).mean()
    rmse = err.pow(2).mean(dim=0).sqrt()
    z = (err / (0.5 * logvar).exp()).std(dim=0)          # calibration: want ~1
    return float(nll), rmse.cpu().numpy(), z.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--epochs", type=int, default=N_EPOCHS)
    ap.add_argument("--n-collect", type=int, default=N_COLLECT)
    ap.add_argument("--n-val", type=int, default=N_VAL)
    ap.add_argument("--goal-weight", type=int, default=1,
                    help="1=goal-weighted sampling (CLAUDE.md), 0=uniform")
    ap.add_argument("--floor", type=float, default=FLOOR)
    ap.add_argument("--d0", type=float, default=D0)
    ap.add_argument("--save-dir", type=str, default=str(SAVE_DIR))
    ap.add_argument("--kl-budget", type=float, default=KL_BUDGET,
                    help="set beta so beta*KL_init/N equals this (arch-independent)")
    ap.add_argument("--hid-size", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=4,
                    help="TOTAL layers incl. output; 4 = 3 hidden")
    ap.add_argument("--beta-nll", type=float, default=0.5,
                    help="beta-NLL (Seitzer et al. 2022); 0=plain Gaussian NLL")
    ap.add_argument("--out-scaler", type=int, default=1,
                    help="1=standardize the 11 target dims (strongly recommended)")
    ap.add_argument("--sd-floor-frac", type=float, default=1e-3,
                    help="floor the output scaler's per-dim std at this fraction "
                         "of the largest std.  Reacher's two target-position dims "
                         "have delta identically 0, so their std is exactly 0 and "
                         "standardizing would divide by ~0; the floor also keeps "
                         "the online surprise z-score for those dims finite.")
    ap.add_argument("--gear", type=float, default=1.0,
                    help="actuator gear MULTIPLIER to pretrain ON (1.0 = shipped)")
    ap.add_argument("--mass", type=float, default=1.0, help="link mass multiplier")
    ap.add_argument("--damping", type=float, default=1.0, help="joint damping mult")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-clip", type=float, default=5.0,
                    help="global grad-norm clip; 0 disables (see pretrain_ant.py)")
    ap.add_argument("--lr-final-frac", type=float, default=0.01,
                    help="cosine-anneal the LR to this fraction of --lr (0 = off)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    t0 = time.time()
    print(f"Collecting {args.n_collect} train + {args.n_val} val transitions on "
          f"Reacher (gear x{args.gear}, mass x{args.mass}, damping x{args.damping})...")
    data = collect_data(args.n_collect, np.random.default_rng(args.seed),
                        gear=args.gear, mass=args.mass, damping=args.damping)
    val = collect_data(args.n_val, np.random.default_rng(args.seed + 10_000),
                       gear=args.gear, mass=args.mass, damping=args.damping)
    print(f"  collected in {time.time() - t0:.1f}s: {len(data)} train, {len(val)} val")

    obs_np, act_np, nxt_np, rew_np = to_arrays(data)
    v_obs, v_act, v_nxt, v_rew = to_arrays(val)
    d = np.linalg.norm(obs_np[:, DIST_IDX], axis=1)
    print(f"  coverage: dist {d.min():.4f}..{d.max():.4f} (mean {d.mean():.4f}) "
          f"| d<0.01: {100 * np.mean(d < 0.01):.1f}%  d<0.03: {100 * np.mean(d < 0.03):.1f}%"
          f"  d<0.05: {100 * np.mean(d < 0.05):.1f}% "
          f"| |qvel| p99 {np.percentile(np.abs(obs_np[:, QVEL_IDX]), 99):.1f} "
          f"| target r {np.linalg.norm(obs_np[:, TARGET_IDX], axis=1).max():.3f} "
          f"| reward {rew_np.min():.2f}..{rew_np.max():.2f}")

    bnn, dyn = make_gaussian_bnn(OBS_DIM, ACT_DIM, hid_size=args.hid_size,
                                 num_layers=args.num_layers)
    bnn.beta_nll = args.beta_nll
    n_par = sum(p.numel() for p in bnn.parameters())
    print(f"  arch: hid={args.hid_size} num_layers={args.num_layers} "
          f"({args.num_layers - 1} hidden) | {n_par:,} params | beta_nll={args.beta_nll}")

    bnn.beta = args.beta
    bnn.num_train_points = len(data)
    bnn.anchor_prior_to_current()
    if args.kl_budget is not None:
        with torch.no_grad():
            kl0 = float(bnn._total_kl().item())
        bnn.beta = args.kl_budget * len(data) / max(kl0, 1e-9)
        print(f"  kl-budget={args.kl_budget}: KL_init={kl0:.3e} N={len(data)} "
              f"-> beta={bnn.beta:.6g}")

    # Input normalizer (mbrl) -- must be fitted on the TRAIN split only.
    dyn.update_normalizer(TransitionBatch(
        obs=obs_np, act=act_np, next_obs=nxt_np, rewards=rew_np,
        dones=np.zeros(len(data), bool)))

    # Train on the SAME representation used at planning time: the NORMALIZED
    # (obs, act) that dyn._get_model_input produces.  Targets stay raw:
    # [next_obs - obs, reward]  (target_is_delta=True, learned_rewards=True).
    def model_in(o, a):
        with torch.no_grad():
            return dyn._get_model_input(torch.tensor(o, device=device),
                                        torch.tensor(a, device=device))
    xs_t = model_in(obs_np, act_np)
    v_xs_t = model_in(v_obs, v_act)
    ys = np.concatenate([nxt_np - obs_np, rew_np[:, None]], axis=1).astype(np.float32)
    v_ys = np.concatenate([v_nxt - v_obs, v_rew[:, None]], axis=1).astype(np.float32)
    ys_t = torch.tensor(ys, device=device); v_ys_t = torch.tensor(v_ys, device=device)

    tgt_sd = ys.std(axis=0)
    if args.out_scaler:
        sd_floor = max(args.sd_floor_frac * tgt_sd.max(), 1e-8)
        sd = np.maximum(tgt_sd, sd_floor)
        floored = np.flatnonzero(tgt_sd < sd_floor)
        bnn.set_output_scaler(ys.mean(axis=0), sd)
        print(f"  output scaler: target std {tgt_sd.min():.4g}..{tgt_sd.max():.4g}"
              f" -> standardized (floor {sd_floor:.3g}"
              f"{f', dims {floored.tolist()} floored' if len(floored) else ''})")

    n = len(data)
    if args.goal_weight:
        gw = goal_weights(obs_np, args.floor, args.d0)
        w = torch.tensor(gw, device=device)
        top = np.mean(np.sort(gw)[-n // 10:]) * n
        print(f"  goal-weighting (d0={args.d0}, floor={args.floor}): "
              f"top-10% states get {top:.2f}x uniform")
    else:
        w = torch.ones(n, device=device)
    n_batches = max(1, n // args.batch_size)
    optimizer = optim.Adam(bnn.parameters(), lr=args.lr)
    sched = (optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.lr * args.lr_final_frac)
             if args.lr_final_frac > 0 else None)

    print(f"Training {args.epochs} epochs x {n_batches} batches "
          f"(batch={args.batch_size}, lr={args.lr})...")
    best = float("inf"); best_state = None; best_epoch = -1; n_skip = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for _ in range(n_batches):
            idx = torch.multinomial(w, args.batch_size, replacement=True)
            optimizer.zero_grad()
            loss, meta = bnn.loss(xs_t[idx], ys_t[idx])
            if not torch.isfinite(loss):
                n_skip += 1
                continue
            loss.backward()
            if args.grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(bnn.parameters(), args.grad_clip)
                if not torch.isfinite(gn):
                    optimizer.zero_grad(); n_skip += 1; continue
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            v_nll, v_rmse, v_z = eval_split(bnn, v_xs_t, v_ys_t)
            if v_nll < best:
                best = v_nll; best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in bnn.state_dict().items()}
            if epoch % 40 == 0 or epoch == args.epochs - 1:
                print(f"  epoch {epoch:3d}: train={epoch_loss / n_batches:8.4f} "
                      f"nll={meta['nll']:8.4f} kl={meta['kl']:.3e} | "
                      f"val_nll={v_nll:8.4f} val_rmse(mean)={v_rmse.mean():.5f} "
                      f"rew_rmse={v_rmse[-1]:.5f} calib_z={v_z.mean():.2f} "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}")
        if sched is not None:
            sched.step()
    if best_state is not None:
        bnn.load_state_dict(best_state)
        print(f"  restored best-val checkpoint (epoch {best_epoch}, val_nll={best:.4f}"
              f"{f', {n_skip} batches skipped' if n_skip else ''})")

    v_nll, v_rmse, v_z = eval_split(bnn, v_xs_t, v_ys_t)
    # NRMSE is meaningless on the two identically-constant target dims (std 0),
    # so average it over the dims that actually vary.
    varying = tgt_sd > 1e-8
    nrmse = v_rmse[varying] / tgt_sd[varying]
    print(f"\nFinal held-out: nll={v_nll:.4f}  mean NRMSE={nrmse.mean():.4f}  "
          f"worst dim={int(np.flatnonzero(varying)[nrmse.argmax()])} "
          f"({nrmse.max():.4f})  calib z={v_z.mean():.2f}")

    bnn.anchor_prior_to_current()
    save_dir = pathlib.Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    dyn.save(str(save_dir))
    bnn.save(str(save_dir), "bnn_dynamics.pth")
    (save_dir / "arch.json").write_text(json.dumps(dict(
        hid_size=args.hid_size, num_layers=args.num_layers, obs_dim=OBS_DIM,
        act_dim=ACT_DIM, gear=args.gear, mass=args.mass, damping=args.damping,
        beta_nll=args.beta_nll, out_scaler=bool(args.out_scaler),
        goal_weight=bool(args.goal_weight), floor=args.floor, d0=args.d0,
        n_collect=args.n_collect, epochs=args.epochs, seed=args.seed,
        val_nll=v_nll, val_nrmse=float(nrmse.mean()))))
    print(f"Saved model + normalizer to {save_dir}  ({time.time() - t0:.0f}s total)")
    print("DONE.")


if __name__ == "__main__":
    main()
