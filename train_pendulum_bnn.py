"""Train Gaussian-head BNN on Pendulum data. Uses sinusoidal exploration for
smoother trajectories, then trains with batched GPU operations.

Saves checkpoint + normalizer to data/pendulum/.
Run:  python train_pendulum_bnn.py
"""

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

N_COLLECT = 10000
BATCH_SIZE = 2048
N_EPOCHS = 300
LR = 1e-3
BETA = 50.0               # KL weight — higher keeps epistemic uncertainty


def collect_data(n_steps=N_COLLECT):
    """Collect transitions using sinusoidal torque for smoother dynamics."""
    env = build_pendulum_env()
    obs, _ = env.reset()
    data = []
    freq = 0.3
    for i in range(n_steps):
        torque = 1.5 * np.sin(freq * i * 0.05) + 0.3 * np.cos(freq * 0.7 * i * 0.05)
        torque = np.clip(torque, -2.0, 2.0)
        act = np.array([torque], dtype=np.float32)
        next_obs, rew, term, trunc, _ = env.step(act)
        data.append((obs.copy(), act.copy(), next_obs.copy(), rew))
        obs = next_obs if not (term or trunc) else env.reset()[0]
    return data


def main():
    print(f"Collecting {N_COLLECT} transitions (sinusoidal exploration)...")
    data = collect_data()
    obs_dim, act_dim = 3, 1
    print(f"  obs_dim={obs_dim} act_dim={act_dim} samples={len(data)}")

    bnn, dyn = make_gaussian_bnn(obs_dim, act_dim)
    bnn.beta = BETA
    bnn.num_train_points = len(data)
    bnn.anchor_prior_to_current()

    # Populate normalizer.
    batch = TransitionBatch(
        obs=np.stack([d[0] for d in data]).astype(np.float32),
        act=np.stack([d[1] for d in data]).astype(np.float32),
        next_obs=np.stack([d[2] for d in data]).astype(np.float32),
        rewards=np.array([d[3] for d in data], dtype=np.float32),
        dones=np.zeros(len(data), dtype=bool),
    )
    dyn.update_normalizer(batch)

    # Prepare training tensors on GPU.
    xs = np.stack([np.concatenate([d[0], d[1]]).astype(np.float32) for d in data])
    ys = np.stack([np.concatenate([d[2] - d[0], [d[3]]]).astype(np.float32) for d in data])
    xs_t = torch.tensor(xs, device=device)
    ys_t = torch.tensor(ys, device=device)
    n = len(data)

    optimizer = optim.Adam(bnn.parameters(), lr=LR)

    print(f"Training {N_EPOCHS} epochs (batch={BATCH_SIZE}, beta={BETA})...")
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss, meta = bnn.loss(xs_t[idx], ys_t[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if epoch % 30 == 0:
            print(f"  epoch {epoch:3d}: loss={epoch_loss / max(n_batches, 1):.4f}  "
                  f"nll={meta['nll']:.4f}  kl={meta['kl']:.1f}")

    bnn.anchor_prior_to_current()
    dyn.save(str(SAVE_DIR))
    print(f"Saved model + normalizer to {SAVE_DIR}")
    print("DONE.")


if __name__ == "__main__":
    main()
