"""Train BNN on oracle-CEM-collected Pendulum transitions.

Run after collect_oracle_data.py completes.
"""
import pathlib, numpy as np, torch
from torch import optim

from config import device
from bnn.gaussian_model import make_gaussian_bnn
from mbrl.types import TransitionBatch

_HERE = pathlib.Path(__file__).parent
DATA_DIR = _HERE / "data" / "pendulum_oracle"
SAVE_DIR = DATA_DIR
SAVE_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 2048
N_EPOCHS = 300
LR = 1e-3
BETA = 50.0

def main():
    data = np.load(DATA_DIR / "transitions.npz")
    obs = data["obs"]; act = data["act"]; next_obs = data["next_obs"]; rew = data["reward"]
    n = len(obs)
    print(f"Loaded {n} oracle transitions")

    bnn, dyn = make_gaussian_bnn(3, 1)
    bnn.beta = BETA
    bnn.num_train_points = n
    bnn.anchor_prior_to_current()

    # Populate input normalizer
    batch = TransitionBatch(obs=obs, act=act, next_obs=next_obs,
                            rewards=rew, dones=np.zeros(n, dtype=bool))
    dyn.update_normalizer(batch)

    # Precompute tensors on GPU
    model_in = np.concatenate([obs, act], axis=-1).astype(np.float32)
    target = np.concatenate([next_obs - obs, rew[:, None]], axis=-1).astype(np.float32)
    xs_t = torch.tensor(model_in, device=device)
    ys_t = torch.tensor(target, device=device)

    optimiser = optim.Adam(bnn.parameters(), lr=LR)
    print(f"Training {N_EPOCHS} epochs...")
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0; nb = 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            optimiser.zero_grad()
            loss, meta = bnn.loss(xs_t[idx], ys_t[idx])
            loss.backward(); optimiser.step()
            epoch_loss += loss.item(); nb += 1
        if epoch % 30 == 0:
            print(f"  epoch {epoch:3d}: loss={epoch_loss/max(nb,1):.3f}  nll={meta['nll']:.3f}")

    bnn.anchor_prior_to_current()
    dyn.save(str(SAVE_DIR))
    bnn.save(str(SAVE_DIR), "bnn_dynamics.pth")
    print(f"Saved to {SAVE_DIR}")
    print("DONE.")

if __name__ == "__main__":
    main()
