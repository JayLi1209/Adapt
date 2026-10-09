"""Quality audit of the pretrained Reacher BNN world model (data/reacher).

Headline question: how much return does the learned world model give up against
an ORACLE that plans with the true simulator, on the env it was pretrained on?

Because both planners are the same CEM -- same horizon, candidates, iterations,
elite fraction, warm start, discount -- the gap between them is attributable to
world-model error and nothing else.  Random actions anchor the bottom of the
scale, so the gap can also be quoted as a fraction of the achievable range:

    model quality = (R_bnn - R_random) / (R_oracle - R_random)

Four axes, as in test_ant_pretrain_quality.py, because a model can pass any one
and still be useless to the planner:

  1. ONE-STEP ACCURACY   per-dim RMSE / NRMSE / R^2 on a FRESH held-out split
     (a different collection seed from the one pretrain_reacher.py early-stopped
     on, so this is not the number the checkpoint was selected against).
  2. CALIBRATION         the Gaussian sigma has to MEAN something: the online
     loop's surprise score is a Mahalanobis z, so a mis-scaled sigma silently
     mis-triggers forget/inflate.  std of (y-mu)/sigma (want ~1) and the
     empirical 1-/2-sigma coverage (want 68 / 95%).
  3. MULTI-STEP ROLLOUT  compounding error over an H-step self-fed rollout,
     against the real MuJoCo trajectory under identical actions and against a
     persistence baseline.
  4. PLANNING            CEM (+CVaR) on the real Reacher, vs the oracle and vs
     random actions.  This is the end-to-end check and the reported result.

Axes 1/2 are broken out GOAL-REGION vs off-goal to verify the CLAUDE.md
goal-prioritised pretraining bought accuracy where the planner lives.

Run:  python3 test_reacher_pretrain_quality.py
      python3 test_reacher_pretrain_quality.py --quick
"""
import argparse
import json
import pathlib

import numpy as np
import torch

from config import device
from bnn import make_gaussian_bnn, load_arch
from env.reacher import (build_reacher_env, reacher_reward, project_reacher_obs,
                         fingertip_xy, OBS_DIM, ACT_DIM, DIST_IDX, EP_LEN)
from planning.continuous_cem import ContinuousCEMAgent
from planning.oracle_cem import MuJoCoOracleCEM, get_state
from pretrain_reacher import collect_data, to_arrays

_HERE = pathlib.Path(__file__).parent

# Semantic blocks of the 11 target dims ([delta_obs(10), reward]).
BLOCKS = [("cos(theta)", slice(0, 2)), ("sin(theta)", slice(2, 4)),
          ("target xy", slice(4, 6)), ("joint vel", slice(6, 8)),
          ("fingertip-target", slice(8, 10)), ("reward", slice(10, 11))]
GOAL_DIST = 0.03          # "on target" radius defining the goal region


def load_model(model_dir, n_train_layers=1):
    hid, nl = load_arch(model_dir)
    bnn, dyn = make_gaussian_bnn(OBS_DIM, ACT_DIM, n_train_layers=n_train_layers,
                                 hid_size=hid, num_layers=nl)
    dyn.input_normalizer.load(model_dir)
    bnn.load(model_dir, "bnn_dynamics.pth")
    bnn.eval()
    return bnn, dyn, hid, nl


def model_input(dyn, obs, act):
    with torch.no_grad():
        return dyn._get_model_input(torch.tensor(obs, device=device),
                                    torch.tensor(act, device=device))


# ── 1 + 2: one-step accuracy and calibration ─────────────────────────────────
@torch.no_grad()
def one_step_report(bnn, xs, ys, tag, per_dim=True):
    mean, logvar = bnn._run_network(xs, sample=False)
    sigma = (0.5 * logvar).exp()
    err = ys - mean
    rmse = err.pow(2).mean(0).sqrt().cpu().numpy()
    sd = ys.std(0).cpu().numpy()
    # dims 4:5 (the target position) have delta identically 0: NRMSE and R^2 are
    # 0/0 there and are excluded from the aggregates rather than reported as nan.
    varying = sd > 1e-8
    nrmse = np.where(varying, rmse / np.maximum(sd, 1e-12), np.nan)
    r2 = (1.0 - (err.pow(2).sum(0) / (ys - ys.mean(0)).pow(2).sum(0).clamp_min(1e-12)
                 ).clamp_min(0)).cpu().numpy()
    r2 = np.where(varying, r2, np.nan)
    z = err / sigma
    z_std = z.std(0).cpu().numpy()
    cov1 = (z.abs() < 1.0).float().mean(0).cpu().numpy()
    cov2 = (z.abs() < 2.0).float().mean(0).cpu().numpy()
    print(f"\n  [{tag}]  n={len(ys)}")
    if per_dim:
        print(f"    {'block':<18}{'RMSE':>10}{'NRMSE':>9}{'R^2':>8}"
              f"{'z_std':>8}{'cov1s':>8}{'cov2s':>8}")
        for name, sl in BLOCKS:
            # The target-xy block is all-nan by construction (constant target ->
            # NRMSE/R^2 are 0/0); nanmean over it warns, so report it as "--".
            blk_nrmse = ("     --" if np.all(np.isnan(nrmse[sl]))
                         else f"{np.nanmean(nrmse[sl]):>9.4f}")
            blk_r2 = ("      --" if np.all(np.isnan(r2[sl]))
                      else f"{np.nanmean(r2[sl]):>8.4f}")
            print(f"    {name:<18}{rmse[sl].mean():>10.5f}"
                  f"{blk_nrmse:>9}{blk_r2:>8}"
                  f"{z_std[sl].mean():>8.2f}{100 * cov1[sl].mean():>7.1f}%"
                  f"{100 * cov2[sl].mean():>7.1f}%")
    print(f"    {'ALL (varying dims)':<18}{rmse[varying].mean():>10.5f}"
          f"{np.nanmean(nrmse):>9.4f}{np.nanmean(r2):>8.4f}"
          f"{z_std[varying].mean():>8.2f}{100 * cov1[varying].mean():>7.1f}%"
          f"{100 * cov2[varying].mean():>7.1f}%")
    return dict(nrmse=float(np.nanmean(nrmse)), r2=float(np.nanmean(r2)),
                z_std=float(z_std[varying].mean()),
                cov1=float(cov1[varying].mean()), cov2=float(cov2[varying].mean()),
                rmse_reward=float(rmse[-1]), rmse_dist=float(rmse[8:10].mean()))


# ── 2b: where does the reward channel's error come from? ─────────────────────
@torch.no_grad()
def reward_head_report(bnn, dyn, obs, act, rew):
    """Decompose the reward channel, because the planner's gap lives here.

    Reacher's reward is r_t = -||d_{t+1}|| - ||a_t||^2, i.e. it depends on the
    fingertip-target vector AFTER the transition.  Three ways to get it, all
    scored against the env's true r_t on the same held-out data:

      (a) LEARNED HEAD    -- the model's 11th output channel, trained directly on
          (s_t, a_t) -> r_t.  It must implicitly do the one-step prediction and
          the norm inside a single regression output.
      (b) ANALYTIC on the model's PREDICTED next state -- take the model's own
          delta prediction for dims 8:10, then apply the closed-form reward.
          This reuses the dynamics the model is already good at.
      (c) ANALYTIC on the CURRENT state -- what reward_fn=reacher_reward gives
          the planner, carrying the documented one-step offset.

    The comparison matters because the SAME network supplies (a) and (b): if (b)
    is much sharper than (a), the deficit is specific to the reward head rather
    than to the model's knowledge of the dynamics.
    """
    xs = model_input(dyn, obs, act)
    mean, _ = bnn._run_network(xs, sample=False)
    mean = mean.cpu().numpy()
    ctrl = (act ** 2).sum(axis=1)
    d_next_pred = obs[:, DIST_IDX] + mean[:, 8:10]        # predicted next d-vector
    r_learned = mean[:, -1]
    r_pred_next = -np.linalg.norm(d_next_pred, axis=1) - ctrl
    r_cur = -np.linalg.norm(obs[:, DIST_IDX], axis=1) - ctrl
    print("\n  reward-channel decomposition (RMSE vs the env's true reward)")
    rows = {}
    for name, r in [("(a) learned head", r_learned),
                    ("(b) analytic on model's predicted next state", r_pred_next),
                    ("(c) analytic on current state (1-step offset)", r_cur)]:
        e = float(np.sqrt(np.mean((r - rew) ** 2)))
        rows[name] = e
        print(f"    {name:<46} RMSE {e:.5f}")
    print(f"    -> the goal-region signal the planner must resolve is the last "
          f"~{GOAL_DIST:.2f} of distance;\n       a reward RMSE above that "
          f"cannot rank a near-target candidate against an on-target one.")
    return rows


# ── 3: multi-step open-loop rollout ──────────────────────────────────────────
def rollout_report(bnn, dyn, horizon, n_traj, seed, project=True, act_rho=0.7):
    """Compare model self-rollouts against real MuJoCo under identical actions."""
    from pretrain_reacher import _sample_state
    rng = np.random.default_rng(seed)
    env = build_reacher_env(max_episode_steps=10 ** 9)
    env.reset(seed=int(rng.integers(1 << 30)))

    starts, acts_all, true_all = [], [], []
    for _ in range(n_traj):
        # Start in the NEAR-goal region: this measures the error the planner
        # actually accumulates while closing on the target, not the error while
        # the arm flails somewhere irrelevant.
        obs = env.set_state_obs(*_sample_state(env, rng, "near"))
        a = np.zeros(ACT_DIM, dtype=np.float32)
        traj, acts = [], []
        for _t in range(horizon):
            eps = rng.uniform(-1, 1, size=ACT_DIM).astype(np.float32)
            a = np.clip(act_rho * a + np.sqrt(1 - act_rho ** 2) * eps, -1, 1
                        ).astype(np.float32)
            nxt, _r, _te, _tr, _ = env.step(a)
            acts.append(a.copy()); traj.append(nxt.copy())
        starts.append(obs.copy()); acts_all.append(acts); true_all.append(traj)

    starts = np.stack(starts).astype(np.float32)
    acts_all = np.array(acts_all, dtype=np.float32)
    true_all = np.array(true_all, dtype=np.float32)

    # Model rollout, mirroring planning.continuous_cem.act's inner loop exactly.
    with torch.no_grad():
        obs_t = torch.tensor(starts, device=device)
        state = dyn.reset(obs_t)
        pred = []
        for t in range(horizon):
            act_t = torch.tensor(acts_all[:, t, :], device=device)
            nxt, _rew, _d, _s = dyn.sample(act_t, state, deterministic=True)
            if project:
                nxt = project_reacher_obs(nxt)
            pred.append(nxt.cpu().numpy())
            obs_t = nxt
            state = dyn.reset(obs_t)
    pred = np.stack(pred, axis=1)

    persist = np.repeat(starts[:, None, :], horizon, axis=1)
    print(f"\n  open-loop rollout error vs real MuJoCo  ({n_traj} traj, "
          f"near-goal starts, projection={'ON' if project else 'OFF'})")
    print(f"    {'H':>4}{'model RMSE':>13}{'persist RMSE':>15}{'ratio':>8}"
          f"{'dist-vec err':>14}")
    rows = {}
    for h in [1, 2, 5, 10, 15, 20, 25, 30]:
        if h > horizon:
            continue
        m = np.sqrt(np.mean((pred[:, h - 1] - true_all[:, h - 1]) ** 2))
        p = np.sqrt(np.mean((persist[:, h - 1] - true_all[:, h - 1]) ** 2))
        de = np.sqrt(np.mean((pred[:, h - 1, DIST_IDX]
                              - true_all[:, h - 1, DIST_IDX]) ** 2))
        print(f"    {h:>4}{m:>13.5f}{p:>15.5f}{m / max(p, 1e-9):>8.3f}{de:>14.5f}")
        rows[h] = dict(model=float(m), persist=float(p),
                       ratio=float(m / max(p, 1e-9)), dist_err=float(de))
    return rows


# ── 4: planning ──────────────────────────────────────────────────────────────
def _episode_stats(returns, finals, last10):
    n = len(returns)
    return dict(ret=float(np.mean(returns)),
                ret_se=float(np.std(returns) / max(np.sqrt(n), 1)),
                final_dist=float(np.mean(finals)),
                final_dist_se=float(np.std(finals) / max(np.sqrt(n), 1)),
                last10_dist=float(np.mean(last10)), n_trials=n)


def _report(label, out):
    print(f"    {label:<42} return {out['ret']:8.3f} +/- {out['ret_se']:.3f}   "
          f"final dist {out['final_dist']:.4f} +/- {out['final_dist_se']:.4f}   "
          f"last-10 dist {out['last10_dist']:.4f}")


def plan_eval_bnn(bnn, dyn, label, n_trials, ep_len, horizon, candidates, iters,
                  k_models, cvar_alpha, gamma, seed, reward_fn=None,
                  aleatoric=False, project=True, reward_clip=None, verbose=False):
    """CEM + CVaR on the real Reacher with the LEARNED model.

    aleatoric=False drops the observation-noise injection from the planner's
    self-fed rollouts (bnn.aleatoric_in_rollout).  MuJoCo Reacher is
    DETERMINISTIC, so that channel is absorbed model error rather than real
    process noise; re-injecting it at every one of H steps simulates randomness
    the plant does not have.  The K posterior draws still differ, so CVaR still
    averages a genuine (epistemic) tail -- see bnn/gaussian_model.py.
    """
    saved = bnn.aleatoric_in_rollout
    bnn.aleatoric_in_rollout = aleatoric
    returns, finals, last10 = [], [], []
    for tr in range(n_trials):
        env = build_reacher_env()
        obs, _ = env.reset(seed=seed + tr)
        agent = ContinuousCEMAgent(
            dyn, bnn, OBS_DIM, ACT_DIM, horizon=horizon, n_cem_iters=iters,
            n_candidates=candidates, k_models=k_models, cvar_alpha=cvar_alpha,
            gamma=gamma, action_low=-1.0, action_high=1.0, reward_fn=reward_fn,
            reward_clip=reward_clip,
            obs_project=project_reacher_obs if project else None,
            rng=np.random.default_rng(seed + tr))
        agent.reset()
        R, ds = 0.0, []
        for _t in range(ep_len):
            a = agent.act(obs)
            obs, r, term, trunc, _ = env.step(a)
            R += r; ds.append(float(np.linalg.norm(obs[DIST_IDX])))
            if term:
                break
        returns.append(R); finals.append(ds[-1]); last10.append(np.mean(ds[-10:]))
        if verbose:
            print(f"      trial {tr + 1}/{n_trials}: return={R:7.3f} "
                  f"final_dist={ds[-1]:.4f}")
    bnn.aleatoric_in_rollout = saved
    out = _episode_stats(returns, finals, last10); out["label"] = label
    _report(label, out)
    return out


def _oracle_reward_fn(nq):
    """Per-step reward from POST-transition simulator states -- exactly the env's
    own reward, r_t = -||fingertip(s_{t+1}) - target|| - ||a_t||^2, with the
    fingertip recovered from qpos by the two-link forward kinematics."""
    def f(states, acts):
        qpos = states[:, :, 1:1 + nq]
        fx, fy = fingertip_xy(qpos[:, :, 0], qpos[:, :, 1])
        d = np.sqrt((fx - qpos[:, :, 2]) ** 2 + (fy - qpos[:, :, 3]) ** 2)
        return -d - (acts ** 2).sum(axis=2)
    return f


def plan_eval_oracle(label, n_trials, ep_len, horizon, candidates, iters, gamma,
                     seed, nthread=12, verbose=False):
    """CEM with the TRUE simulator -- the upper bound on what this CEM can do."""
    returns, finals, last10 = [], [], []
    for tr in range(n_trials):
        env = build_reacher_env()
        obs, _ = env.reset(seed=seed + tr)
        u = env.unwrapped
        agent = MuJoCoOracleCEM(u.model, u.frame_skip, _oracle_reward_fn(u.model.nq),
                                ACT_DIM, horizon=horizon, n_cem_iters=iters,
                                n_candidates=candidates, gamma=gamma,
                                rng=np.random.default_rng(seed + tr), nthread=nthread)
        agent.reset()
        R, ds = 0.0, []
        for _t in range(ep_len):
            a = agent.act(get_state(env))
            obs, r, term, trunc, _ = env.step(a)
            R += r; ds.append(float(np.linalg.norm(obs[DIST_IDX])))
            if term:
                break
        returns.append(R); finals.append(ds[-1]); last10.append(np.mean(ds[-10:]))
        if verbose:
            print(f"      trial {tr + 1}/{n_trials}: return={R:7.3f} "
                  f"final_dist={ds[-1]:.4f}")
    out = _episode_stats(returns, finals, last10); out["label"] = label
    _report(label, out)
    return out


def plan_eval_baseline(label, n_trials, ep_len, seed, mode="random"):
    returns, finals, last10 = [], [], []
    for tr in range(n_trials):
        env = build_reacher_env()
        obs, _ = env.reset(seed=seed + tr)
        rng = np.random.default_rng(seed + tr)
        R, ds = 0.0, []
        for _t in range(ep_len):
            a = rng.uniform(-1, 1, ACT_DIM) if mode == "random" else np.zeros(ACT_DIM)
            obs, r, term, trunc, _ = env.step(a)
            R += r; ds.append(float(np.linalg.norm(obs[DIST_IDX])))
            if term:
                break
        returns.append(R); finals.append(ds[-1]); last10.append(np.mean(ds[-10:]))
    out = _episode_stats(returns, finals, last10); out["label"] = label
    _report(label, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, default=str(_HERE / "data" / "reacher"))
    ap.add_argument("--n-test", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--rollout-h", type=int, default=30)
    ap.add_argument("--rollout-traj", type=int, default=300)
    ap.add_argument("--plan-trials", type=int, default=100,
                    help="CLAUDE.md default: 100 trials with error bars")
    ap.add_argument("--plan-len", type=int, default=EP_LEN,
                    help="Reacher-v5's native episode length; the task IS 50 steps")
    ap.add_argument("--plan-h", type=int, default=15)
    ap.add_argument("--plan-candidates", type=int, default=500)
    ap.add_argument("--plan-iters", type=int, default=8)
    ap.add_argument("--k-models", type=int, default=3,
                    help="posterior draws; >1 makes CVaR meaningful (CLAUDE.md)")
    ap.add_argument("--cvar-alpha", type=float, default=0.8)
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="CLAUDE.md: no discount")
    ap.add_argument("--nthread", type=int, default=12, help="oracle rollout threads")
    ap.add_argument("--skip-planning", action="store_true")
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--skip-ablations", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        given = {a.lstrip('-').replace('-', '_') for a in __import__('sys').argv[1:]
                 if a.startswith('--')}
        for k, v in dict(n_test=8000, rollout_traj=100, plan_trials=5,
                         plan_candidates=200, plan_iters=3).items():
            if k not in given:
                setattr(args, k, v)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    bnn, dyn, hid, nl = load_model(args.model_dir)
    meta = {}
    f = pathlib.Path(args.model_dir) / "arch.json"
    if f.exists():
        meta = json.loads(f.read_text())
    print("\n=== Reacher BNN pretraining audit ===")
    print(f"  model_dir : {args.model_dir}")
    print(f"  arch      : hid={hid} num_layers={nl} "
          f"({sum(p.numel() for p in bnn.parameters()):,} params) "
          f"beta_nll={meta.get('beta_nll')} out_scaler={meta.get('out_scaler')}")
    print(f"  pretrained on: gear x{meta.get('gear')} mass x{meta.get('mass')} "
          f"damping x{meta.get('damping')} | {meta.get('n_collect')} transitions, "
          f"{meta.get('epochs')} epochs")

    print(f"\nCollecting {args.n_test} FRESH test transitions (seed {args.seed})...")
    data = collect_data(args.n_test, np.random.default_rng(args.seed))
    obs, act, nxt, rew = to_arrays(data)
    xs = model_input(dyn, obs, act)
    ys = torch.tensor(np.concatenate([nxt - obs, rew[:, None]], axis=1
                                     ).astype(np.float32), device=device)

    print("\n--- 1+2. one-step accuracy & calibration ---")
    overall = one_step_report(bnn, xs, ys, "ALL test states")
    d = np.linalg.norm(obs[:, DIST_IDX], axis=1)
    goal = d < GOAL_DIST
    gi = torch.tensor(np.flatnonzero(goal), device=device)
    oi = torch.tensor(np.flatnonzero(~goal), device=device)
    print(f"\n  goal-region states (dist < {GOAL_DIST}): {goal.sum()} / {len(goal)} "
          f"({100 * goal.mean():.1f}%)")
    goal_m = one_step_report(bnn, xs[gi], ys[gi], "GOAL region", per_dim=False)
    off_m = one_step_report(bnn, xs[oi], ys[oi], "off-goal", per_dim=False)

    rew_decomp = reward_head_report(bnn, dyn, obs, act, rew)

    print("\n--- 3. multi-step open-loop rollout ---")
    roll = rollout_report(bnn, dyn, args.rollout_h, args.rollout_traj, args.seed,
                          project=True)
    roll_np = rollout_report(bnn, dyn, args.rollout_h, args.rollout_traj, args.seed,
                             project=False)

    plans = {}
    if not args.skip_planning:
        print("\n--- 4. planning on the real Reacher ---")
        print(f"    SHARED CEM config: H={args.plan_h} J={args.plan_candidates} "
              f"iters={args.plan_iters} gamma={args.gamma} "
              f"| {args.plan_trials} trials x {args.plan_len} steps")
        print(f"    BNN planner adds:  K={args.k_models} posterior draws, "
              f"CVaR alpha={args.cvar_alpha} (CLAUDE.md: cVaR + CEM)")
        plans["random"] = plan_eval_baseline("random actions", args.plan_trials,
                                             args.plan_len, args.seed, "random")
        plans["zero"] = plan_eval_baseline("zero action (arm holds still)",
                                           args.plan_trials, args.plan_len,
                                           args.seed, "zero")
        plans["bnn"] = plan_eval_bnn(
            bnn, dyn, "BNN CEM+CVaR, learned reward [DEFAULT]", args.plan_trials,
            args.plan_len, args.plan_h, args.plan_candidates, args.plan_iters,
            args.k_models, args.cvar_alpha, args.gamma, args.seed)
        if not args.skip_ablations:
            plans["bnn_alea"] = plan_eval_bnn(
                bnn, dyn, "  ablation: + aleatoric noise in rollout",
                args.plan_trials, args.plan_len, args.plan_h, args.plan_candidates,
                args.plan_iters, args.k_models, args.cvar_alpha, args.gamma,
                args.seed, aleatoric=True)
            plans["bnn_noproj"] = plan_eval_bnn(
                bnn, dyn, "  ablation: no manifold projection", args.plan_trials,
                args.plan_len, args.plan_h, args.plan_candidates, args.plan_iters,
                args.k_models, args.cvar_alpha, args.gamma, args.seed, project=False)
            plans["bnn_analytic"] = plan_eval_bnn(
                bnn, dyn, "  ablation: analytic reward (isolates reward head)",
                args.plan_trials, args.plan_len, args.plan_h, args.plan_candidates,
                args.plan_iters, args.k_models, args.cvar_alpha, args.gamma,
                args.seed, reward_fn=reacher_reward)
        if not args.skip_oracle:
            plans["oracle"] = plan_eval_oracle(
                "ORACLE CEM (true MuJoCo dynamics)", args.plan_trials,
                args.plan_len, args.plan_h, args.plan_candidates, args.plan_iters,
                args.gamma, args.seed, nthread=args.nthread)

    # ── verdict ──────────────────────────────────────────────────────────────
    print("\n=== VERDICT ===")
    checks = [
        ("1-step NRMSE < 0.30", overall["nrmse"] < 0.30, f"{overall['nrmse']:.4f}"),
        ("1-step R^2 > 0.90", overall["r2"] > 0.90, f"{overall['r2']:.4f}"),
        ("calibration z_std in [0.5,2]", 0.5 < overall["z_std"] < 2.0,
         f"{overall['z_std']:.2f}"),
        ("goal region no worse than off-goal (NRMSE)",
         goal_m["nrmse"] <= off_m["nrmse"] * 1.1,
         f"goal {goal_m['nrmse']:.4f} vs off {off_m['nrmse']:.4f}"),
    ]
    if 20 in roll:
        checks.append(("H=20 rollout beats persistence", roll[20]["ratio"] < 1.0,
                       f"ratio {roll[20]['ratio']:.3f}"))
    gap = None
    if "bnn" in plans:
        checks.append(("BNN CEM beats random actions",
                       plans["bnn"]["ret"] > plans["random"]["ret"],
                       f"{plans['bnn']['ret']:.2f} vs {plans['random']['ret']:.2f}"))
    if "oracle" in plans and "bnn" in plans:
        hi = plans["oracle"]["ret"]
        r_bnn = plans["bnn"]["ret"]
        lo_rand, lo_zero = plans["random"]["ret"], plans["zero"]["ret"]
        best_lbl, best = max(
            ((k, v) for k, v in plans.items() if k.startswith("bnn")),
            key=lambda kv: kv[1]["ret"])
        # Two floors, because they say different things.  Random actions is the
        # conventional baseline but a FLATTERING one here: random torques mostly
        # burn control cost (-||a||^2 averages ~-0.67/step), so most of the
        # oracle-over-random range is won just by not thrashing the actuators.
        # Zero action -- the arm holds still and pays only the distance term --
        # is the honest floor for "did the planner actually reach the target".
        frac = (r_bnn - lo_rand) / max(hi - lo_rand, 1e-9)
        frac_z = (r_bnn - lo_zero) / max(hi - lo_zero, 1e-9)
        gap = dict(oracle=hi, bnn=r_bnn, random=lo_rand, zero=lo_zero,
                   frac_of_range=float(frac), frac_of_range_vs_zero=float(frac_z),
                   deficit=float(hi - r_bnn),
                   best_bnn_label=best["label"], best_bnn=best["ret"])
        print(f"  ORACLE GAP: oracle {hi:.3f}  |  BNN {r_bnn:.3f}  |  "
              f"zero-action {lo_zero:.3f}  |  random {lo_rand:.3f}")
        print(f"    deficit vs oracle = {hi - r_bnn:.3f} return "
              f"({100 * (hi - r_bnn) / abs(hi):.1f}% of |oracle|)")
        print(f"    BNN recovers {100 * frac:.1f}% of the oracle-over-RANDOM range, "
              f"{100 * frac_z:.1f}% of the oracle-over-ZERO-ACTION range")
        if best_lbl != "bnn":
            print(f"    NOTE: best BNN variant is '{best['label'].strip()}' "
                  f"at {best['ret']:.3f}, ahead of the default by "
                  f"{best['ret'] - r_bnn:.3f}")
        checks.append(("BNN within 25% of oracle-over-zero-action range",
                       frac_z > 0.75, f"{100 * frac_z:.1f}%"))
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<45} {detail}")
    print(f"\n  {sum(ok for _, ok, _ in checks)}/{len(checks)} checks passed")

    out = pathlib.Path(args.out or (pathlib.Path(args.model_dir) / "audit.json"))
    out.write_text(json.dumps(dict(
        overall=overall, goal=goal_m, off_goal=off_m, rollout=roll,
        rollout_noproj=roll_np, reward_decomp=rew_decomp,
        planning=plans, oracle_gap=gap,
        config=vars(args), checks={n: bool(o) for n, o, _ in checks}), indent=2))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
