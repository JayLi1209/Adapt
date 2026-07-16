"""Train Gaussian BNN on Lunar Lander data collected via random exploration.
Saves checkpoint + normalizer to data/lunar_lander/.

Run:  python train_lunar_lander_bnn.py
"""

import pathlib
import numpy as np
import torch
from torch import optim

from config import device
from bnn.gaussian_model import BayesianDynamicsModel
from env.lunar_lander import build_lunar_lander_env
from mbrl.types import TransitionBatch

_HERE = pathlib.Path(__file__).parent
SAVE_DIR = _HERE / "data" / "lunar_lander"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

N_COLLECT = 20000
BATCH_SIZE = 2048
N_EPOCHS = 300
LR = 1e-3
BETA = 50.0


def collect_data(n_steps=N_COLLECT):
    """Collect transitions using random actions on stationary Lunar Lander."""
    env = build_lunar_lander_env()
    obs, _ = env.reset()
    data = []
    for i in range(n_steps):
        act = env.action_space.sample()
        next_obs, rew, term, trunc, _ = env.step(act)
        data.append((obs.copy(), np.float32(act), next_obs.copy(), rew))
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{n_steps} transitions")
    return data


def main():
    print(f"Collecting {N_COLLECT} transitions (random exploration)...")
    data = collect_data()
    obs_dim, act_dim = 8, 1
    n = len(data)
    print(f"  obs_dim={obs_dim} act_dim={act_dim} samples={n}")

    bnn = BayesianDynamicsModel(
        in_size=obs_dim + act_dim, out_size=obs_dim + 1,
        device=device, hid_size=256, num_layers=3,
        prior_std=1.0, beta=BETA, num_mc_samples=3, num_weight_groups=1,
    )
    bnn.num_train_points = n
    bnn.anchor_prior_to_current()

    # Populate normalizer
    batch = TransitionBatch(
        obs=np.stack([d[0] for d in data]).astype(np.float32),
        act=np.stack([d[1] for d in data]).astype(np.float32).reshape(-1, 1),
        next_obs=np.stack([d[2] for d in data]).astype(np.float32),
        rewards=np.array([d[3] for d in data], dtype=np.float32),
        dones=np.zeros(n, dtype=bool),
    )

    import mbrl.models as models
    dyn = models.OneDTransitionRewardModel(
        bnn, target_is_delta=True, normalize=True, learned_rewards=True
    )
    dyn.update_normalizer(batch)

    # Prepare tensors
    xs = np.stack([np.concatenate([d[0], [d[1]]]).astype(np.float32) for d in data])
    ys = np.stack([np.concatenate([d[2] - d[0], [d[3]]]).astype(np.float32) for d in data])
    xs_t = torch.tensor(xs, device=device)
    ys_t = torch.tensor(ys, device=device)

    optimizer = optim.Adam(bnn.parameters(), lr=LR)
    print(f"Training {N_EPOCHS} epochs (batch={BATCH_SIZE}, beta={BETA})...")
    best_nll = float("inf")
    best_state = None
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(n, device=device)
        epoch_loss, epoch_nll, nb = 0.0, 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            optimizer.zero_grad()
            loss, meta = bnn.loss(xs_t[idx], ys_t[idx])
            loss.backward(); optimizer.step()
            epoch_loss += loss.item(); epoch_nll += meta["nll"]; nb += 1
        avg_nll = epoch_nll / max(nb, 1)
        if avg_nll < best_nll:
            best_nll = avg_nll
            best_state = {k: v.clone() for k, v in bnn.state_dict().items()}
        if epoch % 30 == 0:
            print(f"  epoch {epoch:3d}: loss={epoch_loss/max(nb,1):.3f}  nll={avg_nll:.3f}")

    if best_state is not None:
        bnn.load_state_dict(best_state)
        print(f"Restored best model (nll={best_nll:.4f})")

    bnn.anchor_prior_to_current()
    dyn.save(str(SAVE_DIR))
    bnn.save(str(SAVE_DIR), "bnn_dynamics.pth")
    print(f"Saved to {SAVE_DIR}")
    print("DONE.")


if __name__ == "__main__":
    main()
