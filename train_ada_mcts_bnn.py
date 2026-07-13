"""Pretrain latent BNN on STATIONARY Pendulum (mass=1.0) using oracle CEM data.
Saves checkpoint + frozen trunk to data/pendulum_ada/.

Run:  python train_ada_mcts_bnn.py
"""

import pathlib, numpy as np, torch
from torch import optim

from config import device
from bnn.latent_model import make_latent_bnn, LATENT_DIM
from mbrl.types import TransitionBatch

_HERE = pathlib.Path(__file__).parent
SAVE_DIR = _HERE / "data" / "pendulum_ada"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Collect stationary data using oracle CEM ─────────────────────────────────────
from oracle_cem_baseline import PendulumSim

N_EPISODES = 20
EPISODE_LEN = 100
BATCH_SIZE = 2048
N_EPOCHS = 500
LR = 1e-3
BETA = 50.0


def _to_obs(th, thdot):
    return np.array([np.cos(th), np.sin(th), thdot], dtype=np.float32)


def collect_stationary_data(rng):
    """Oracle CEM rollouts with mass=1.0 ONLY (stationary)."""
    from oracle_cem_baseline import cem_act, H, I, J
    sim = PendulumSim()
    sim.set_mass(1.0)
    data = []
    for ep in range(N_EPISODES):
        s = sim.reset(seed=ep)
        for _ in range(EPISODE_LEN):
            obs = _to_obs(s[0], s[1])
            torque = cem_act(sim, s.copy(), rng)
            ns, rew = sim.step(s, torque)
            next_obs = _to_obs(ns[0], ns[1])
            data.append((obs, np.array([torque], dtype=np.float32), next_obs, np.float32(rew)))
            s = ns
        print(f"  episode {ep+1}/{N_EPISODES} ({len(data)} transitions)")
    return data


def main():
    rng = np.random.default_rng(0)

    print(f"Collecting {N_EPISODES} stationary episodes (mass=1.0)...")
    data = collect_stationary_data(rng)
    n = len(data); obs_dim, act_dim = 3, 1
    print(f"  {n} transitions collected")

    bnn, dyn = make_latent_bnn(obs_dim, act_dim, LATENT_DIM)
    bnn.beta = BETA
    bnn.num_train_points = n
    bnn.anchor_prior_to_current()

    # Populate normalizer
    batch = TransitionBatch(
        obs=np.stack([d[0] for d in data]).astype(np.float32),
        act=np.stack([d[1] for d in data]).astype(np.float32),
        next_obs=np.stack([d[2] for d in data]).astype(np.float32),
        rewards=np.array([d[3] for d in data], dtype=np.float32),
        dones=np.zeros(n, dtype=bool),
    )
    dyn.update_normalizer(batch)

    # Prepare GPU tensors: model_in = (obs, act) WITHOUT latent (latent added inside BNN)
    xs = np.stack([np.concatenate([d[0], d[1]]).astype(np.float32) for d in data])
    ys = np.stack([np.concatenate([d[2] - d[0], [d[3]]]).astype(np.float32) for d in data])
    xs_t = torch.tensor(xs, device=device)
    ys_t = torch.tensor(ys, device=device)

    opt = optim.Adam(bnn.parameters(), lr=LR)
    best_nll = float("inf")
    best_state = None
    print(f"Training {N_EPOCHS} epochs (batch={BATCH_SIZE}, beta={BETA})...")
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(n, device=device)
        epoch_loss, epoch_nll, nb = 0.0, 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            opt.zero_grad()
            loss, meta = bnn.loss(xs_t[idx], ys_t[idx])
            loss.backward(); opt.step()
            epoch_loss += loss.item()
            epoch_nll += meta["nll"]
            nb += 1
        avg_nll = epoch_nll / max(nb, 1)
        if avg_nll < best_nll:
            best_nll = avg_nll
            best_state = {k: v.clone() for k, v in bnn.state_dict().items()}
        if epoch % 50 == 0:
            print(f"  epoch {epoch:3d}: loss={epoch_loss/max(nb,1):.3f}  nll={avg_nll:.3f}")

    # Restore best checkpoint
    if best_state is not None:
        bnn.load_state_dict(best_state)
        print(f"Restored best model (nll={best_nll:.4f})")

    bnn.anchor_prior_to_current()

    # Save full checkpoint
    dyn.save(str(SAVE_DIR))
    bnn.save_latent(str(SAVE_DIR))
    bnn.save(str(SAVE_DIR), "bnn_dynamics.pth")

    # Save a FROZEN-TRUNK copy: freeze hidden layers, save as "pretrained trunk"
    bnn.freeze_trunk()
    bnn.save_latent(str(SAVE_DIR), "latent_bnn_frozen.pth")
    print(f"Saved to {SAVE_DIR}")
    print(f"  Latent vector: {bnn.latent.data.cpu().numpy()}")
    print("DONE.")


if __name__ == "__main__":
    main()
