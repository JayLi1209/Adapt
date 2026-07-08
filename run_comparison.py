"""Pipeline: collect oracle data → train BNN → run both methods → compare.

Usage: python run_comparison.py [--skip-collect] [--skip-train]
"""

import copy, math, pathlib, sys, time, numpy as np, torch

_HERE = pathlib.Path(__file__).parent
BNN_DIR = _HERE / "data" / "pendulum_oracle"
BNN_DIR.mkdir(parents=True, exist_ok=True)

from config import device, ETA, GAMMA_UNCERTAINTY
from env import build_pendulum_env
from drift import DriftFilterV2
from bnn import make_gaussian_bnn, surprise_gaussian, forget_gaussian, mean_sigma
from planning.continuous_cem import ContinuousCEMAgent, H_PLAN, N_CEM_ITERS, N_CANDIDATES, K_MODELS, GAMMA

MASS_SCHEDULE = [(0, 1.0), (80, 3.0)]
CHANGE_STEPS = [80]
N_TRIALS = 10
TRIAL_LEN = 100
K_FORGET = 5

# ── Step 1: Collect oracle data ─────────────────────────────────────────────────
def step_collect():
    print("[1/4] Collecting oracle CEM transitions...")
    from collect_oracle_data import collect; collect()

# ── Step 2: Train BNN ───────────────────────────────────────────────────────────
def step_train():
    print("[2/4] Training BNN on oracle data...")
    from train_bnn_oracle import main; main()

# ── Step 3: Our method (BNN + CEM + forget) ─────────────────────────────────────
def run_ours():
    print("[3/4] Running OUR method (BNN + CEM + forget)...")
    env = build_pendulum_env(MASS_SCHEDULE)
    bnn, dyn = make_gaussian_bnn(3, 1)
    dyn.load(str(BNN_DIR))
    bnn.num_weight_groups = 1
    init_state = copy.deepcopy(bnn.state_dict())

    agent = ContinuousCEMAgent(dyn, bnn, 3, 1, device=device,
                               horizon=H_PLAN, n_cem_iters=N_CEM_ITERS,
                               n_candidates=N_CANDIDATES, k_models=K_MODELS, gamma=GAMMA)
    returns = []
    for trial in range(N_TRIALS):
        obs, _ = env.reset(); agent.reset(); bnn.load_state_dict(init_state)
        drift = DriftFilterV2(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)
        total, post = 0.0, 0
        for step in range(TRIAL_LEN):
            if step in CHANGE_STEPS: drift.reset(); agent.notify_change()
            action = agent.act(obs)
            next_obs, rew, term, trunc, _ = env.step(action)
            total += float(rew)
            act_arr = np.asarray(action, dtype=np.float32).ravel()
            vs = surprise_gaussian(dyn, bnn, obs, act_arr, next_obs, rew)
            drift.update(min(vs["nu2"], 50.0))
            if step >= CHANGE_STEPS[0]:
                post += 1
                if post % K_FORGET == 0:
                    forget_gaussian(bnn, drift, inflate_mode="additive", q_max=0.1)
            obs = next_obs
            if term or trunc: break
        returns.append(total)
        print(f"  Ours trial {trial+1}: {total:.1f}")
    return returns

# ── Step 4: ADA-MCTS baseline ───────────────────────────────────────────────────
def run_ada_mcts():
    print("[4/4] Running ADA-MCTS baseline...")
    from ada_mcts_pendulum import mcts_act, TORQUES, N_ACTIONS, MCTS_SIMS
    env = build_pendulum_env(MASS_SCHEDULE)
    bnn, dyn = make_gaussian_bnn(3, 1)
    dyn.load(str(BNN_DIR))
    bnn.num_weight_groups = 1
    init_state = copy.deepcopy(bnn.state_dict())

    returns = []
    for trial in range(N_TRIALS):
        obs, _ = env.reset(); bnn.load_state_dict(init_state)
        total = 0.0; t0 = time.time()
        for step in range(TRIAL_LEN):
            a_idx, torque = mcts_act(dyn, bnn, obs)
            obs, rew, term, trunc, _ = env.step(np.array([torque], dtype=np.float32))
            total += float(rew)
            if term or trunc: break
        returns.append(total)
        print(f"  ADA-MCTS trial {trial+1}: {total:.1f} ({time.time()-t0:.0f}s)")
    return returns

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    skip_collect = "--skip-collect" in sys.argv
    skip_train = "--skip-train" in sys.argv

    if not skip_collect: step_collect()
    if not skip_train: step_train()

    print(f"\n{'='*60}")
    print(f"COMPARISON: Our method vs ADA-MCTS on Pendulum")
    print(f"  {N_TRIALS} trials x {TRIAL_LEN} steps, mass {MASS_SCHEDULE}")
    print(f"{'='*60}")

    r_ours = run_ours()
    r_mcts = run_ada_mcts()

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Ours (CEM+forget):  avg={np.mean(r_ours):.1f}  std={np.std(r_ours):.1f}  min={np.min(r_ours):.1f}  max={np.max(r_ours):.1f}")
    print(f"  ADA-MCTS:           avg={np.mean(r_mcts):.1f}  std={np.std(r_mcts):.1f}  min={np.min(r_mcts):.1f}  max={np.max(r_mcts):.1f}")
    print(f"{'='*60}")

    with open(str(_HERE / "comparison_results.md"), "w") as f:
        f.write("# Pendulum Comparison Results\n\n")
        f.write(f"| Method | Avg | Std | Min | Max |\n")
        f.write(f"|---|---|---|---|---|\n")
        f.write(f"| Ours (CEM+forget) | {np.mean(r_ours):.1f} | {np.std(r_ours):.1f} | {np.min(r_ours):.1f} | {np.max(r_ours):.1f} |\n")
        f.write(f"| ADA-MCTS | {np.mean(r_mcts):.1f} | {np.std(r_mcts):.1f} | {np.min(r_mcts):.1f} | {np.max(r_mcts):.1f} |\n")
    print("Saved to comparison_results.md")

if __name__ == "__main__":
    main()
