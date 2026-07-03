"""Train the Gaussian-head BNN on Pendulum data collected via random exploration.

Uses OneDTransitionRewardModel.update_normalizer + save so that both the BNN
checkpoint and the input/output normalizer stats are persisted in the format
that dyn.load() expects.

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

N_COLLECT = 8000          # random transitions to collect
BATCH_SIZE = 1024
N_EPOCHS = 200
LR = 1e-3
BETA = 100.0              # KL weight (higher = keep more epistemic uncertainty)


def collect_and_normalize(bnn, dyn, n_steps=N_COLLECT):
    """Collect random data and populate the input normalizer."""
    env = build_pendulum_env()
    obs, _ = env.reset()
    data = []
    for _ in range(n_steps):
        act = np.random.uniform(-2.0, 2.0, size=(1,)).astype(np.float32)
        next_obs, rew, term, trunc, _ = env.step(act)
        data.append((obs.copy(), act.copy(), next_obs.copy(), rew))
        if term or trunc:
            obs, _ = env.reset()
        else:
            obs = next_obs

    obs_dim = data[0][0].shape[0]
    act_dim = data[0][1].shape[0]

    batch = TransitionBatch(
        obs=np.stack([d[0] for d in data]).astype(np.float32),
        act=np.stack([d[1] for d in data]).astype(np.float32),
        next_obs=np.stack([d[2] for d in data]).astype(np.float32),
        rewards=np.array([d[3] for d in data], dtype=np.float32),
        dones=np.zeros(len(data), dtype=bool),
    )
    dyn.update_normalizer(batch)
    return data, obs_dim, act_dim


def make_training_batches(data, batch_size=BATCH_SIZE):
    """Yield (model_in, target) mini-batches on device."""
    xs, ys = [], []
    for obs, act, next_obs, rew in data:
        model_in = np.concatenate([obs, act]).astype(np.float32)
        target = np.concatenate([next_obs - obs, [rew]]).astype(np.float32)
        xs.append(model_in)
        ys.append(target)
    xs = np.stack(xs)
    ys = np.stack(ys)
    n = xs.shape[0]
    idx = np.random.permutation(n)
    for i in range(0, n, batch_size):
        batch_idx = idx[i : i + batch_size]
        yield (torch.tensor(xs[batch_idx], device=device),
               torch.tensor(ys[batch_idx], device=device))


def main():
    print(f"Collecting {N_COLLECT} random transitions from Pendulum...")
    bnn, dyn = make_gaussian_bnn(3, 1)       # obs_dim=3, act_dim=1 (pendulum)
    data, obs_dim, act_dim = collect_and_normalize(bnn, dyn)
    print(f"  obs_dim={obs_dim} act_dim={act_dim}")

    bnn.beta = BETA               # higher KL weight keeps epistemic uncertainty
    bnn.num_train_points = len(data)
    bnn.anchor_prior_to_current()

    optimizer = optim.Adam(bnn.parameters(), lr=LR)

    print(f"Training {N_EPOCHS} epochs on {len(data)} transitions (batch={BATCH_SIZE})...")
    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for model_in, target in make_training_batches(data):
            optimizer.zero_grad()
            loss, meta = bnn.loss(model_in, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if epoch % 20 == 0:
            avg = epoch_loss / max(n_batches, 1)
            print(f"  epoch {epoch:3d}: loss={avg:.6f}  nll={meta['nll']:.4f}  kl={meta['kl']:.4f}")

    bnn.anchor_prior_to_current()

    # OneDTransitionRewardModel.save saves model.pth (norm stats) + BNN checkpoint.
    dyn.save(str(SAVE_DIR))
    print(f"Saved model + normalizer to {SAVE_DIR}")
    print("DONE.")


if __name__ == "__main__":
    main()
