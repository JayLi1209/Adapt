"""Pretrain the Gaussian-head BNN on the DEFAULT Pendulum (mass=1, g=10).

Same collection/training pipeline as train_pendulum_bnn.py, but minibatches are
drawn with a GOAL-WEIGHTED sampler so transitions near the upright goal get
trained more (per CLAUDE.md: "pretraining should prioritize the transitions
close to the goal more").  The goal weight of a transition is

    w = ((cos_theta + 1) / 2) ** POS_POWER * exp(-0.5 * (theta_dot / VEL_SCALE)^2) + FLOOR

i.e. large when the pole is up (cos_theta -> 1) and slow (theta_dot -> 0).  The
FLOOR keeps every sample reachable so off-goal dynamics are still learned.

Saves checkpoint + normalizer to data/pendulum/ (the same files run_continuous /
sweep_pendulum_alpha load).  Reuses env + model code unchanged.

Run:  python pretrain_pendulum.py
"""
import argparse
import json
import pathlib

import numpy as np
import torch
from torch import optim

from config import device
from bnn.gaussian_model import make_gaussian_bnn
from env.pendulum import build_pendulum_env
from mbrl.types import TransitionBatch

_HERE = pathlib.Path(__file__).parent
SAVE_DIR = _HERE / "data" / "pendulum"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

N_COLLECT = 20000
BATCH_SIZE = 2048
N_EPOCHS = 300
LR = 1e-3
# KL weight in loss = nll + BETA*kl/num_train_points.  BETA=50 makes KL ~99.8% of
# the loss (kl~1.7e5, N=1e4) so the net never fits -> use a small BETA so the mean
# predictions are accurate while keeping some epistemic spread for forget.
BETA = 1.0

# ── goal-weighting (CLAUDE.md) ────────────────────────────────────────────────
POS_POWER = 2.0           # emphasis on being upright (cos_theta near +1)
VEL_SCALE = 2.0           # rad/s scale; smaller -> stronger low-speed emphasis
FLOOR = 0.05              # min sampling weight so off-goal dynamics still train

# ── reward-target scaling ─────────────────────────────────────────────────────
# The model's 4th output is the reward.  On the RAW Pendulum scale the reward
# lives in [-16.2736, 0] while the delta-obs targets live in ~[-0.4, 0.4], so the
# reward channel dominates the 4-dim Gaussian NLL and the shared trunk spends its
# capacity fitting reward instead of dynamics.  "unit" maps reward affinely onto
# [0, 1] so all four output channels have comparable scale.
R_MIN = -(np.pi ** 2 + 0.1 * 8.0 ** 2 + 0.001 * 2.0 ** 2)   # -16.2736
R_MAX = 0.0


def scale_reward(r, mode):
    """raw: unchanged.  unit: [R_MIN, 0] -> [0, 1] (higher still = better)."""
    return r if mode == "raw" else (r - R_MIN) / (R_MAX - R_MIN)


def collect_data(n_steps=N_COLLECT, rng=None, mass=1.0, grav=10.0):
    """Collect DEFAULT-env transitions with broad state coverage.

    Slow sinusoidal exploration alone never swings the pole up, so the model
    never sees the goal region and mispredicts it.  We periodically TELEPORT the
    pendulum to a fresh state -- half the teleports land NEAR UPRIGHT (theta~0,
    the goal), half anywhere -- so the dataset covers the balance region densely
    (which goal-weighting then further emphasises, per CLAUDE.md).  Torque mixes a
    sinusoid with noise for action diversity.
    """
    rng = rng or np.random.default_rng(0)
    # pretrain on whatever (mass, gravity) this checkpoint is meant to model;
    # the schedule is constant so the whole dataset comes from one dynamics
    env = build_pendulum_env([(0, mass)], [(0, grav)])
    obs, _ = env.reset()
    data = []
    for i in range(n_steps):
        if i % 25 == 0:                      # teleport to diversify coverage
            if rng.random() < 0.5:           # near upright (the goal)
                th, thd = rng.normal(0.0, 0.35), rng.normal(0.0, 0.8)
            else:                            # anywhere on the circle
                th, thd = rng.uniform(-np.pi, np.pi), rng.uniform(-4.0, 4.0)
            env.unwrapped.state = np.array([th, thd], dtype=np.float64)
            obs = np.array([np.cos(th), np.sin(th), thd], dtype=np.float32)
        torque = np.clip(1.2 * np.sin(0.05 * i) + rng.normal(0.0, 0.9), -2.0, 2.0)
        act = np.array([torque], dtype=np.float32)
        next_obs, rew, term, trunc, _ = env.step(act)
        data.append((obs.copy(), act.copy(), next_obs.copy(), rew))
        obs = next_obs if not (term or trunc) else env.reset()[0]
    return data


def load_demos(path, n_demo):
    """Oracle-demo transitions (successful swing-up + balance) from
    collect_oracle_demos.py.  These supply the coordinated ACTION sequences that
    collect_data's sinusoid+noise torque never produces.  Returns [] if absent."""
    p = pathlib.Path(path)
    if n_demo <= 0 or not p.exists():
        return []
    d = np.load(p)
    n = min(n_demo, len(d["obs"]))
    idx = np.random.default_rng(0).choice(len(d["obs"]), size=n, replace=False)
    return [(d["obs"][i], d["act"][i], d["next_obs"][i], float(d["reward"][i]))
            for i in idx]


def goal_weights(obs_arr, floor=FLOOR):
    """Per-sample sampling weights emphasising near-upright, low-speed states.

    `floor` matters more than it looks: with floor=0.05 an upright/slow state gets
    ~1.05 while a hanging-down fast state gets ~0.05, a 21x penalty on the
    swing-up regime the planner must traverse.  Raising the floor keeps the
    CLAUDE.md goal emphasis while not starving mid-swing dynamics.
    """
    cos_theta = obs_arr[:, 0]
    theta_dot = obs_arr[:, 2]
    upright = ((cos_theta + 1.0) / 2.0) ** POS_POWER
    slow = np.exp(-0.5 * (theta_dot / VEL_SCALE) ** 2)
    w = upright * slow + floor
    return (w / w.sum()).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--epochs", type=int, default=N_EPOCHS)
    ap.add_argument("--n-collect", type=int, default=N_COLLECT)
    ap.add_argument("--goal-weight", type=int, default=1,
                    help="1=goal-weighted sampling (CLAUDE.md), 0=uniform")
    ap.add_argument("--floor", type=float, default=FLOOR,
                    help="goal-weight floor; higher = less starvation of mid-swing")
    ap.add_argument("--n-oracle", type=int, default=0,
                    help="oracle-demo transitions to mix in (0=off)")
    ap.add_argument("--demo-npz", type=str,
                    default=str(_HERE / "data" / "pendulum_demos.npz"))
    ap.add_argument("--save-dir", type=str, default=str(SAVE_DIR),
                    help="where to write the checkpoint (default: data/pendulum)")
    ap.add_argument("--kl-budget", type=float, default=None,
                    help="set beta so that beta*KL_init/N equals this, making the KL "
                         "regulariser ARCHITECTURE-INDEPENDENT.  KL grows with the "
                         "parameter count, so a bigger net with a fixed beta is "
                         "over-regularised and underfits (2x256: KL/N~7.8; 4x512: "
                         "KL/N~100).  Use ~7.8 to match the shipped 2x256 net.")
    ap.add_argument("--hid-size", type=int, default=256,
                    help="trunk width (shipped: 256)")
    ap.add_argument("--num-layers", type=int, default=3,
                    help="TOTAL layers incl. output; num_layers-1 hidden. "
                         "shipped 3 = 2 hidden; use 5 for 4 hidden layers")
    ap.add_argument("--reward-scale", choices=["raw", "unit"], default="raw",
                    help="target scale of the model's reward channel: raw = the "
                         "native [-16.27, 0]; unit = mapped to [0, 1] so it does "
                         "not dominate the 4-dim NLL against the delta-obs dims")
    ap.add_argument("--mass", type=float, default=1.0,
                    help="pendulum mass to pretrain ON (the model's prior belief). "
                         "The shipped checkpoint is mass=1.")
    ap.add_argument("--grav", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for data collection + torch init (fix it to compare "
                         "reward scales with everything else held identical)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Collecting {args.n_collect} transitions on Pendulum "
          f"(mass={args.mass}, g={args.grav})...")
    data = collect_data(args.n_collect, rng=np.random.default_rng(args.seed),
                        mass=args.mass, grav=args.grav)
    demos = load_demos(args.demo_npz, args.n_oracle)
    if demos:
        print(f"  + {len(demos)} ORACLE-DEMO transitions from {args.demo_npz}")
        data = data + demos
    obs_dim, act_dim = 3, 1
    print(f"  obs_dim={obs_dim} act_dim={act_dim} samples={len(data)} "
          f"beta={args.beta} epochs={args.epochs} goal_weight={args.goal_weight}")

    bnn, dyn = make_gaussian_bnn(obs_dim, act_dim,
                                 hid_size=args.hid_size, num_layers=args.num_layers)
    n_par = sum(p.numel() for p in bnn.parameters())
    print(f'  arch: hid={args.hid_size} num_layers={args.num_layers} ({args.num_layers-1} hidden) | {n_par:,} params')
    bnn.beta = args.beta
    bnn.num_train_points = len(data)
    bnn.anchor_prior_to_current()

    # Architecture-independent KL weighting: pick beta from the KL right after the
    # prior is anchored, so depth/width changes do not silently change how hard the
    # posterior is pulled toward the prior.
    if args.kl_budget is not None:
        with torch.no_grad():
            kl0 = float(bnn._total_kl().item())
        bnn.beta = args.kl_budget * len(data) / max(kl0, 1e-9)
        print(f"  kl-budget={args.kl_budget}: KL_init={kl0:.3e} N={len(data)} "
              f"-> beta={bnn.beta:.6f} (was {args.beta})")

    obs_np = np.stack([d[0] for d in data]).astype(np.float32)
    batch = TransitionBatch(
        obs=obs_np,
        act=np.stack([d[1] for d in data]).astype(np.float32),
        next_obs=np.stack([d[2] for d in data]).astype(np.float32),
        rewards=np.array([d[3] for d in data], dtype=np.float32),
        terminateds=np.zeros(len(data), dtype=bool),
        truncateds=np.zeros(len(data), dtype=bool),
    )
    dyn.update_normalizer(batch)

    # IMPORTANT: train on the SAME representation used at inference/planning --
    # the NORMALIZED (obs,act) input that dyn._get_model_input produces (this is
    # also what retrain_gaussian buffers).  Training on raw concat(obs,act) while
    # dyn normalizes at inference is a train/serve mismatch that yields garbage
    # predictions.  Target stays raw: [next_obs - obs, reward] (target_is_delta).
    obs_all = torch.tensor(obs_np, device=device)
    act_all = torch.tensor(np.stack([d[1] for d in data]).astype(np.float32), device=device)
    with torch.no_grad():
        xs_t = dyn._get_model_input(obs_all, act_all)
    ys = np.stack([np.concatenate(
        [d[2] - d[0], [scale_reward(d[3], args.reward_scale)]]).astype(np.float32)
        for d in data])
    print(f"  reward target scale = {args.reward_scale} "
          f"(range {ys[:, -1].min():.3f}..{ys[:, -1].max():.3f}; "
          f"delta-obs range {ys[:, :3].min():.3f}..{ys[:, :3].max():.3f})")
    ys_t = torch.tensor(ys, device=device)
    n = len(data)

    # Sampling distribution over the dataset: goal-weighted (CLAUDE.md) or uniform.
    if args.goal_weight:
        w = torch.tensor(goal_weights(obs_np, args.floor), device=device)
        print(f"  goal-weighting (floor={args.floor}): top-10% upright states get "
              f"{np.mean(np.sort(goal_weights(obs_np, args.floor))[-n // 10:]) * n:.2f}x uniform")
    else:
        w = torch.ones(n, device=device)
    n_batches = max(1, n // BATCH_SIZE)

    optimizer = optim.Adam(bnn.parameters(), lr=LR)

    print(f"Training {args.epochs} epochs (batch={BATCH_SIZE}, beta={args.beta})...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for _ in range(n_batches):
            # Draw a minibatch from the (goal-weighted) distribution, with replacement.
            idx = torch.multinomial(w, BATCH_SIZE, replacement=True)
            optimizer.zero_grad()
            loss, meta = bnn.loss(xs_t[idx], ys_t[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 30 == 0:
            print(f"  epoch {epoch:3d}: loss={epoch_loss / n_batches:.4f}  "
                  f"nll={meta['nll']:.4f}  kl={meta['kl']:.1f}")

    bnn.anchor_prior_to_current()
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dyn.save(str(save_dir))
    bnn.save(str(save_dir), "bnn_dynamics.pth")
    (save_dir / "reward_scale.json").write_text(json.dumps(dict(
        reward_scale=args.reward_scale, r_min=R_MIN, r_max=R_MAX, seed=args.seed)))
    (save_dir / "arch.json").write_text(json.dumps(dict(
        hid_size=args.hid_size, num_layers=args.num_layers,
        mass=args.mass, grav=args.grav)))
    print(f"Saved model + normalizer to {save_dir} (beta={args.beta}, "
          f"goal_weight={args.goal_weight}, floor={args.floor}, "
          f"n_oracle={args.n_oracle})")
    print("DONE.")


if __name__ == "__main__":
    main()
