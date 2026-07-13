"""Our method with latent BNN: CEM + surprise/forget + online finetuning on Pendulum.
Uses the same pretrained latent BNN as ADA-MCTS for fair comparison.

Online finetuning of head+latent during non-stationary adaptation, consistent
with the discrete FrozenLake approach.

Run:  python run_continuous_latent.py
"""

import copy, pathlib
import numpy as np
import torch
from torch import optim

from config import device, ETA, GAMMA_UNCERTAINTY
from env import build_pendulum_env
from drift import DriftFilterV2
from bnn import surprise_gaussian, forget_gaussian, mean_sigma
from bnn.latent_model import make_latent_bnn, LATENT_DIM
from planning.continuous_cem import ContinuousCEMAgent, H_PLAN, N_CEM_ITERS, N_CANDIDATES, K_MODELS, GAMMA

_HERE = pathlib.Path(__file__).parent
LOG = str(_HERE / "run_pendulum_adapt_latent.log")
BNN_DIR = _HERE / "data" / "pendulum_ada"

MASS_SCHEDULE = [(0, 1.0), (80, 3.0)]
CHANGE_STEPS = [80]
N_TRIALS = 20
TRIAL_LEN = 100
K_FORGET = 5
HEAD_LR = 1e-3
FINETUNE_EVERY = 5
MIN_BUF = 8


def main():
    out = open(LOG, "w")
    def log(*a):
        print(*a, file=out); out.flush()
        print(*a)

    eval_env = build_pendulum_env(MASS_SCHEDULE)
    obs_dim, act_dim = 3, 1

    bnn, dyn = make_latent_bnn(obs_dim, act_dim, LATENT_DIM)
    dyn.load(str(BNN_DIR))
    frozen_path = BNN_DIR / "latent_bnn_frozen.pth"
    if frozen_path.exists():
        bnn.load_latent(str(BNN_DIR), "latent_bnn_frozen.pth")
        log(f"Loaded pretrained latent BNN (frozen trunk) from {BNN_DIR}")
    else:
        log("ERROR: Run train_ada_mcts_bnn.py first")
        return

    bnn.num_weight_groups = 1
    init_state = copy.deepcopy(bnn.state_dict())

    agent = ContinuousCEMAgent(dyn, bnn, obs_dim, act_dim, device=device,
                               horizon=H_PLAN, n_cem_iters=N_CEM_ITERS,
                               n_candidates=N_CANDIDATES,
                               k_models=K_MODELS, gamma=GAMMA)

    log("=" * 80)
    log(f"CEM+BNN+forget+finetune (latent={LATENT_DIM}) | mass={MASS_SCHEDULE} | "
        f"trials={N_TRIALS}x{TRIAL_LEN} | K_forget={K_FORGET} | FT_every={FINETUNE_EVERY}")
    log(f"  H={H_PLAN} I={N_CEM_ITERS} J={N_CANDIDATES} K={K_MODELS} gamma={GAMMA}")
    log("=" * 80)

    returns_hist = []

    for trial in range(N_TRIALS):
        obs, _ = eval_env.reset()
        agent.reset()
        bnn.load_state_dict(init_state)
        drift = DriftFilterV2(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)

        buffer = []
        n_post_change = 0
        opt = optim.Adam(bnn.head_parameters(), lr=HEAD_LR)

        total_return = 0.0
        post = 0

        for step in range(TRIAL_LEN):
            if step in CHANGE_STEPS:
                drift.reset()
                agent.notify_change()
                n_post_change = 0
                buffer.clear()
                log(f"  [CHANGE] t={step}: mass={eval_env.unwrapped.m}, "
                    f"latent={bnn.latent.data.cpu().numpy().round(4)}")

            action = agent.act(obs)
            next_obs, reward, term, trunc, _ = eval_env.step(action)
            total_return += float(reward)

            act_arr = np.asarray(action, dtype=np.float32).ravel()
            vs = surprise_gaussian(dyn, bnn, obs, act_arr, next_obs, reward)
            raw_s = min(vs["nu2"], 50.0)
            drift.update(max(raw_s, 1e-6))

            post_change = step >= CHANGE_STEPS[0]
            if post_change:
                post += 1
                if post % K_FORGET == 0:
                    sig_before = mean_sigma(bnn)
                    applied = forget_gaussian(bnn, drift, inflate_mode="additive",
                                              q_max=0.1)
                    sig_after = mean_sigma(bnn)
                    if applied > 0:
                        log(f"  FORGET t={step}: applied={applied:.4f} "
                            f"sigma {sig_before:.4f}->{sig_after:.4f}")

                # Online finetuning: head + latent only
                n_post_change += 1
                model_in = np.concatenate([obs, act_arr]).astype(np.float32)
                target = np.concatenate([next_obs - obs, [float(reward)]]).astype(np.float32)
                buffer.append((model_in, target))
                if len(buffer) > 100:
                    buffer.pop(0)

                if len(buffer) >= MIN_BUF and n_post_change % FINETUNE_EVERY == 0:
                    batch_size = min(16, len(buffer))
                    idxs = np.random.choice(len(buffer), size=batch_size, replace=False)
                    for i in idxs:
                        mi, tg = buffer[i]
                        mi_t = torch.tensor(mi, device=device).unsqueeze(0)
                        tg_t = torch.tensor(tg, device=device).unsqueeze(0)
                        bnn.num_train_points = max(len(buffer), 1)
                        loss, _ = bnn.loss(mi_t, tg_t)
                        loss.backward()
                    opt.step(); opt.zero_grad()
                    if n_post_change % (FINETUNE_EVERY * 3) == 0:
                        log(f"  [FT] t={step}: n_buf={len(buffer)} "
                            f"latent={bnn.latent.data.cpu().numpy().round(4)}")

            if step % 50 == 0:
                a_val = float(act_arr[0])
                log(f"  t={step:3d} | a={a_val:+.3f} r={reward:+.3f} | "
                    f"nu2={raw_s:.3f} | lam={drift.lambda_hat:+.3f} | "
                    f"sigma={mean_sigma(bnn):.4f}")

            obs = next_obs
            if term or trunc:
                break

        returns_hist.append(total_return)
        log(f"  => TRIAL {trial+1}: return={total_return:.1f} | "
            f"lam={drift.lambda_hat:+.3f} | "
            f"latent_final={bnn.latent.data.cpu().numpy().round(4)}")

    log(f"\n{'='*60}")
    log(f"RESULTS: avg={np.mean(returns_hist):.1f} std={np.std(returns_hist):.1f} "
        f"min={np.min(returns_hist):.1f} max={np.max(returns_hist):.1f}")
    log("DONE.")
    out.close()


if __name__ == "__main__":
    main()
