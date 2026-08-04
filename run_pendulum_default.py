"""DEFAULT-setting Pendulum run (CLAUDE.md) with a per-step return curve.

Default setting per CLAUDE.md:
    mass 1 -> 4 at ts 0, stop at episode 200, discount 0.99 (planner),
    gaussian head, averaged over 100 trials with error bars.

Because the change fires at ts 0 the model is wrong from the very first step and
there is NO pre-change window, so the drift filter cannot learn an empirical
baseline.  We therefore feed the SELF-CALIBRATING delta_n surprise (E[delta_n]=1
under a correct model), which is what the filter's default baseline of 1.0
expects.  Feeding raw nu2 here would pin lambda_hat at ~-1 and forget would
NEVER fire (the silent-no-op footgun documented in sweep_pendulum_alpha.py).

Planning scores rollouts with the ANALYTIC ground-truth Pendulum reward, and the
learned reward channel is excluded from the surprise / retrain signals, so the
model's reward head drives no decision.

Per trial it records the full per-step reward trace.  While running it prints a
progress bar with an ETA and, as each trial finishes, the running average total
return across all finished trials.  Results are saved INCREMENTALLY (after every
trial), so a crash or a Ctrl-C never loses completed work.

Outputs (into --out-dir, default results/pendulum_default):
    <tag>.json  full config + per-trial reward traces (rewritten each trial)
    <tag>.csv   t, mean_cum_return, sem_cum_return, mean_reward, sem_reward
    <tag>.png   the figure: mean +/- 1 s.e.m. vs timestep

Run:
    conda activate nsgym
    CUDA_VISIBLE_DEVICES=<gpu> python run_pendulum_default.py --trials 100

Re-plot later without re-running:
    python run_pendulum_default.py --plot-only --out-dir results/pendulum_default
"""
import argparse
import copy
import hashlib
import json
import pathlib
import time

import numpy as np
import torch

from config import device, ETA, GAMMA_UNCERTAINTY
from env import build_pendulum_env
from drift import DriftFilterV2
from bnn import (make_gaussian_bnn, load_arch, surprise_gaussian, forget_gaussian, mean_sigma,
                 adapter_params_gaussian, trunk_params_gaussian, retrain_gaussian,
                 retrain_layers_gaussian, variational_params_gaussian,
                 retrain_gaussian_elbo)
from planning.continuous_cem import (ContinuousCEMAgent, GAMMA, project_unit_circle,
                                     pendulum_reward)

SAVE_DIR = pathlib.Path(__file__).parent / "data" / "pendulum"

# ── CLAUDE.md default setting ─────────────────────────────────────────────────
OBS_DIM, ACT_DIM = 3, 1
DEFAULT_MASS, DEFAULT_GRAV = 1.0, 10.0
TARGET_MASS = 4.0
CHANGE_STEP = 0            # mass is 4 from the very first step
TRIAL_LEN = 200            # "stop at episode 200"
K_FORGET = 5
INFLATE_MODE, Q_MAX = "additive", 0.1
SURPRISE_STAT = "delta_n"  # REQUIRED at change_step=0 (self-calibrating, E=1)
SURPRISE_CLIP = 50.0
RETRAIN_LR, RETRAIN_STEPS = 1e-3, 5
SURPRISE_USE_REWARD_DIM = False   # planner uses the analytic reward
RETRAIN_USE_REWARD_DIM = False
# True per-step Pendulum reward range: max cost = pi^2 + 0.1*8^2 + 0.001*2^2
REWARD_RANGE = (-(np.pi ** 2 + 0.1 * 8.0 ** 2 + 0.001 * 2.0 ** 2), 0.0)


def fmt_time(sec):
    sec = int(max(0, sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def progress_bar(done, total, elapsed, width=40):
    """Plain 1..N progress bar, e.g.
    [############----------------------------] 12/100 ( 12.0%)  elapsed 18m04s"""
    frac = done / total if total else 0.0
    filled = int(round(width * frac))
    bar = "#" * filled + "-" * (width - filled)
    return (f"[{bar}] {done}/{total} ({frac * 100:5.1f}%)  "
            f"elapsed {fmt_time(elapsed)}")


def predict_mean(dyn, bnn, obs, act):
    """One-step MEAN-model prediction (no weight sampling): the model's best guess
    of the next state and reward, in the trained delta space [delta-obs, reward]."""
    saved = bnn.num_weight_groups
    bnn.num_weight_groups = 1
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        act_t = torch.as_tensor(act, dtype=torch.float32, device=device).unsqueeze(0)
        model_in = dyn._get_model_input(obs_t, act_t)
        mean, _ = bnn._run_network(model_in, sample=False)
    bnn.num_weight_groups = saved
    mean = mean.squeeze(0).cpu().numpy()
    return np.asarray(obs, dtype=np.float64) + mean[:OBS_DIM], float(mean[OBS_DIM])


def run_trial(env, dyn, bnn, init_state, agent, seed, args, log=None):
    """One 200-step episode on the shifted env; returns the per-step reward list.

    With log= supplied, emits the same 3-line-per-step block as diag_mass4_forget
    (state/action/reward/return, model prediction vs truth, surprise/drift/forget).
    Every step is post-change here because the shift fires at ts 0."""
    torch.manual_seed(seed + 10000)
    obs, _ = env.reset(seed=seed)
    bnn.load_state_dict(init_state)           # restart from the pretrained weights
    agent.reset()
    drift = DriftFilterV2(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY,
                          window=args.k_forget)

    # `retrain_layers` are the LAYER objects the ELBO path needs (KL subset +
    # re-anchor target); the mu-only path only ever needs the parameter lists.
    retrain_layers = []
    if not args.retrain:
        retrain_params = []                       # nothing unfrozen (pure forget)
    elif args.unfreeze_trunk > 0:                                          # scope B
        retrain_layers = retrain_layers_gaussian(bnn, n_unfreeze=args.unfreeze_trunk,
                                                 use_adapters=False)
        retrain_params = (variational_params_gaussian(retrain_layers, train_rho=True)
                          if args.retrain_variance
                          else trunk_params_gaussian(bnn, args.unfreeze_trunk))
    else:                                                                  # scope A
        retrain_layers = retrain_layers_gaussian(
            bnn, n_unfreeze=(1 if args.update_final_layer else 0), use_adapters=True)
        retrain_params = (variational_params_gaussian(retrain_layers, train_rho=True)
                          if args.retrain_variance
                          else adapter_params_gaussian(bnn, bool(args.update_final_layer)))
    retrain_opt = (torch.optim.Adam(retrain_params, lr=RETRAIN_LR)
                   if retrain_params else None)
    buf_in, buf_tgt = [], []

    rewards, post = [], 0
    n_forget_fired = 0
    plan_rets = []                            # planner's predicted return each step
    trace = []                                # per-step state/action/reward (oracle-aligned)
    for step in range(TRIAL_LEN):
        if step == CHANGE_STEP:               # fires at t=0: shifted dynamics are live
            drift.reset()
            agent.notify_change()

        mass_now = float(env.unwrapped.m)
        theta = float(np.arctan2(obs[1], obs[0]))
        action = agent.act(obs)
        plan_rets.append(float(agent.last_plan_return))
        plan_ret, plan_ret_best = agent.last_plan_return, agent.last_plan_return_best
        act_arr = np.asarray(action, dtype=np.float32).ravel()
        pred_next, pred_reward = (predict_mean(dyn, bnn, obs, act_arr)
                                  if log else (np.zeros(OBS_DIM), 0.0))
        analytic_r = float(pendulum_reward(
            torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0),
            torch.as_tensor(act_arr, dtype=torch.float32, device=device).unsqueeze(0)
        ).item()) if log else 0.0

        next_obs, reward, term, trunc, _ = env.step(action)
        rewards.append(float(reward))
        cum_return = float(np.sum(rewards))
        state_pred_err = float(np.linalg.norm(pred_next - np.asarray(next_obs, np.float64)))

        # same convention as oracle_mass4.py: post-step state, action, reward
        trace.append(dict(t=step, cos=round(float(next_obs[0]), 5),
                          sin=round(float(next_obs[1]), 5),
                          theta_dot=round(float(next_obs[2]), 5),
                          torque=round(float(act_arr[0]), 5),
                          reward=round(float(reward), 5), mass=mass_now))

        vs = surprise_gaussian(dyn, bnn, obs, act_arr, next_obs, reward,
                               use_reward_dim=args.surprise_reward_dim)
        drift.update(max(min(vs[SURPRISE_STAT], SURPRISE_CLIP), 1e-6))

        q_hat_step, nll_step, sig_before, sig_after = 0.0, None, None, None
        if step >= CHANGE_STEP:
            post += 1
            if retrain_opt is not None:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                act_t = torch.as_tensor(act_arr, dtype=torch.float32, device=device)
                nxt_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    buf_in.append(dyn._get_model_input(obs_t.unsqueeze(0),
                                                       act_t.unsqueeze(0)))
                buf_tgt.append(torch.cat(
                    [nxt_t - obs_t,
                     torch.tensor([reward], dtype=torch.float32, device=device)]
                ).unsqueeze(0))
            if post % args.k_forget == 0:
                if not args.no_forget:
                    sig_before = mean_sigma(bnn)
                    q_hat_step = forget_gaussian(bnn, drift, inflate_mode=INFLATE_MODE,
                                                 q_max=Q_MAX, forget_mean=False)
                    sig_after = mean_sigma(bnn)
                    if q_hat_step > 0:
                        n_forget_fired += 1
                        if args.retrain_variance and retrain_layers:
                            # re-anchor so the ELBO's KL contracts FROM the just-
                            # inflated belief, not from the fixed pretrained prior
                            # (prior_std=1.0), which would push sigma UP toward 1.
                            bnn.anchor_prior_to_current(include_sigma=True,
                                                        layers=retrain_layers)
                if retrain_opt is not None and buf_in:
                    if args.retrain_variance:
                        nll_step, _kl_step, _sig_step = retrain_gaussian_elbo(
                            bnn, retrain_opt, torch.cat(buf_in), torch.cat(buf_tgt),
                            retrain_layers, n_steps=RETRAIN_STEPS,
                            beta=args.elbo_beta,
                            kl_denom=(args.kl_denom if args.kl_denom > 0 else None),
                            use_reward_dim=RETRAIN_USE_REWARD_DIM)
                    else:
                        nll_step = retrain_gaussian(bnn, retrain_opt, torch.cat(buf_in),
                                                    torch.cat(buf_tgt), n_steps=RETRAIN_STEPS,
                                                    use_reward_dim=RETRAIN_USE_REWARD_DIM)

        # ── per-step block, same format as diag_mass4_forget / forget.log ──────
        if log:
            log(f"    [post t={step:3d} m={mass_now:g}] "
                f"s(cos,sin,w,th)=({obs[0]:+.3f},{obs[1]:+.3f},{obs[2]:+6.3f},{theta:+.3f}) "
                f"a={float(act_arr[0]):+.2f} r={reward:+7.3f} R={cum_return:+8.1f}")
            log(f"         pred_ns=({pred_next[0]:+.3f},{pred_next[1]:+.3f},{pred_next[2]:+6.3f}) "
                f"pred_r={pred_reward:+.3f} |Δstate|={state_pred_err:6.3f} "
                f"plan_ret={plan_ret:+9.2f}(best {plan_ret_best:+9.2f}) "
                f"[analytic_r={analytic_r:+.3f} err={analytic_r - float(reward):+.2e}]")
            extra = ""
            if sig_before is not None:
                fired = "FIRED" if q_hat_step > 0 else "deadband"
                extra = (f" | FORGET {fired} q={q_hat_step:.4f} "
                         f"sig {sig_before:.3f}->{sig_after:.3f}")
            if nll_step is not None:
                extra += f" | RETRAIN nll={nll_step:+.3f}(buf={len(buf_in)})"
            log(f"         surp: nu2={vs['nu2']:.5f} dn={vs['delta_n']:.2f} "
                f"epi={vs['epistemic']:.4f} ale={vs['aleatoric']:.4f} | "
                f"drift lam={drift.lambda_hat:+.5f} base={drift.baseline:.5f} "
                f"dbar={drift.delta_bar:.4f}{extra}")

        obs = next_obs
        if term or trunc:
            break
    pr = np.array(plan_rets)
    fin = pr[np.isfinite(pr)]
    plan_stats = dict(min=float(fin.min()) if len(fin) else float("nan"),
                      max=float(fin.max()) if len(fin) else float("nan"),
                      n_nonfinite=int(np.sum(~np.isfinite(pr))))
    return rewards, n_forget_fired, plan_stats, trace


# ── curve aggregation + plotting ──────────────────────────────────────────────
def aggregate(all_rewards):
    """(t, mean/sem of cumulative return, mean/sem of per-step reward)."""
    L = min(len(r) for r in all_rewards)
    R = np.array([r[:L] for r in all_rewards])          # (n_trials, L)
    C = np.cumsum(R, axis=1)
    n = len(R)
    denom = np.sqrt(n) if n > 1 else 1.0
    return dict(
        t=np.arange(1, L + 1),
        mean_cum=C.mean(0), sem_cum=(C.std(0, ddof=1) / denom) if n > 1 else np.zeros(L),
        mean_rew=R.mean(0), sem_rew=(R.std(0, ddof=1) / denom) if n > 1 else np.zeros(L),
        n_trials=n,
    )


def plot_curve(agg, out_png, title_suffix=""):
    """Two stacked panels, one series each (so no legend box -- the title names it):
    cumulative return and per-step reward, each with a +/-1 s.e.m. band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#1a1d21", "#5c6470", "#dfe3e8"
    C_CUM, C_REW = "#2f6fd0", "#0f9d8c"      # one hue per panel, single series each

    t = agg["t"]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True,
                             gridspec_kw=dict(hspace=0.22))

    for ax, (mean, sem, color, ylab, sub) in zip(axes, [
        (agg["mean_cum"], agg["sem_cum"], C_CUM, "Cumulative return",
         "Mean cumulative undiscounted return"),
        (agg["mean_rew"], agg["sem_rew"], C_REW, "Reward per step",
         "Mean per-step reward"),
    ]):
        ax.fill_between(t, mean - sem, mean + sem, color=color, alpha=0.20, linewidth=0)
        ax.plot(t, mean, color=color, linewidth=2.0, solid_capstyle="round")
        ax.set_ylabel(ylab, color=INK, fontsize=10)
        ax.set_title(f"{sub}  (n={agg['n_trials']} trials, band = ±1 s.e.m.)",
                     color=INK, fontsize=11, loc="left", pad=8)
        ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)

    # direct label the final value (selective, not every point)
    axes[0].annotate(f"{agg['mean_cum'][-1]:.1f}",
                     xy=(t[-1], agg["mean_cum"][-1]), xytext=(-6, 8),
                     textcoords="offset points", ha="right",
                     color=INK, fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Timestep", color=INK, fontsize=10)
    axes[0].set_xlim(t[0], t[-1])

    fig.suptitle(f"Pendulum default setting: mass 1→4 at t=0{title_suffix}",
                 color=INK, fontsize=12.5, x=0.02, ha="left", y=0.985)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.90, bottom=0.08, hspace=0.24)
    fig.savefig(out_png, dpi=170, facecolor="white")
    plt.close(fig)


def write_csv(agg, path):
    with open(path, "w") as f:
        f.write("t,mean_cum_return,sem_cum_return,mean_reward,sem_reward\n")
        for i in range(len(agg["t"])):
            f.write(f"{agg['t'][i]},{agg['mean_cum'][i]:.5f},{agg['sem_cum'][i]:.5f},"
                    f"{agg['mean_rew'][i]:.5f},{agg['sem_rew'][i]:.5f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--k-models", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--candidates", type=int, default=500)
    ap.add_argument("--cem-iters", type=int, default=8)
    ap.add_argument("--cvar-alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1000, help="trial t uses seed+t")
    ap.add_argument("--retrain", action="store_true", help="mean adaptation on")
    ap.add_argument("--k-forget", type=int, default=K_FORGET,
                    help="forget+retrain every K post-change steps (pendulum default 5; "
                         "FrozenLake uses 1 = every step). Also sets the drift filter "
                         "window so each firing consumes exactly the new evidence.")
    ap.add_argument("--no-forget", action="store_true", help="ablation: forget off")
    ap.add_argument("--retrain-variance", action="store_true",
                    help="retrain mu AND rho by minimizing the negative ELBO "
                         "(sampled forward + KL) instead of the mu-only NLL. Under "
                         "the mu-only path the forward is sample=False, so rho gets "
                         "exactly zero gradient no matter what is in the optimizer.")
    ap.add_argument("--kl-denom", type=float, default=0.0,
                    help="with --retrain-variance: scale the KL by this instead of "
                         "the buffer size (0 = use the buffer size, the standard "
                         "mean-field NLL + KL/N per-observation update)")
    ap.add_argument("--elbo-beta", type=float, default=1.0,
                    help="with --retrain-variance: weight on the KL term")
    ap.add_argument("--clip-reward", action="store_true",
                    help="clamp every imagined step's reward to the TRUE Pendulum "
                         "range [-16.2736, 0]; bounds the H-step planned return to "
                         "[-535, 0] instead of letting unclipped imagined theta_dot "
                         "blow 0.1*theta_dot^2 up to -1e35 / overflow")
    ap.add_argument("--n-train-layers", type=int, default=1,
                    help="SCOPE A (adapter inject): unfrozen online layers INCL. the "
                         "head. 1=head only; 2=+1 identity adapter; 3=+2 adapters.")
    ap.add_argument("--update-final-layer", type=int, default=1,
                    help="scope A: also retrain the output head's mu (with --retrain)")
    ap.add_argument("--unfreeze-trunk", type=int, default=0,
                    help="SCOPE B (true fine-tune): thaw the top K PRETRAINED layers' "
                         "mu (head=1, +1 hidden=2, ...). >0 overrides scope A.")
    ap.add_argument("--model-dir", default=str(SAVE_DIR),
                    help="pretrained checkpoint dir (default data/pendulum)")
    ap.add_argument("--save-trace", action="store_true",
                    help="store the per-step state/action/reward trace in the JSON "
                         "(same field layout as oracle_mass4.py, for direct diffing)")
    ap.add_argument("--no-step-log", action="store_true",
                    help="suppress the 3-line-per-step block (keep only the "
                         "per-trial summary + progress bar)")
    ap.add_argument("--surprise-reward-dim", action="store_true",
                    help="INCLUDE the learned reward channel in the surprise "
                         "(default excludes it, since planning uses analytic reward)")
    ap.add_argument("--out-dir", default="results/pendulum_default")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--plot-only", action="store_true",
                    help="rebuild the figure from an existing <tag>.json; no runs")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("default" + ("_retrain" if args.retrain else "")
                       + ("_noforget" if args.no_forget else "")
                       + (f"_kf{args.k_forget}" if args.k_forget != K_FORGET else "")
                       + ("_clipr" if args.clip_reward else "")
                       + ("_srdim" if args.surprise_reward_dim else ""))
    json_path, csv_path, png_path = (out_dir / f"{tag}.json", out_dir / f"{tag}.csv",
                                     out_dir / f"{tag}.png")
    log_path = out_dir / f"{tag}.log"

    if args.plot_only:
        d = json.loads(json_path.read_text())
        agg = aggregate([r["rewards"] for r in d["records"]])
        write_csv(agg, csv_path); plot_curve(agg, png_path)
        print(f"replotted {png_path} from {len(d['records'])} trials")
        return

    print(f"PENDULUM DEFAULT SETTING (CLAUDE.md)\n"
          f"  mass {DEFAULT_MASS}->{TARGET_MASS} @ts {CHANGE_STEP} | gravity {DEFAULT_GRAV} | "
          f"episode {TRIAL_LEN} | gamma_plan {GAMMA} | gaussian head\n"
          f"  trials={args.trials} K={args.k_models} alpha={args.cvar_alpha} "
          f"H={args.horizon} J={args.candidates} iters={args.cem_iters}\n"
          f"  forget={not args.no_forget} forget_mean=False retrain={args.retrain} "
          f"surprise={SURPRISE_STAT} k_forget={args.k_forget} planner_reward=analytic "
          f"clip_reward={args.clip_reward}"
          f"{f' {REWARD_RANGE[0]:.4f}..{REWARD_RANGE[1]:g}' if args.clip_reward else ''} "
          f"surprise_reward_dim={args.surprise_reward_dim}\n"
          f"  imagined transitions per decision = "
          f"{args.candidates * args.k_models * args.horizon * args.cem_iters:,}\n",
          flush=True)

    # change at t=0 -> the schedule must be [(0, target)]; [(0,default),(0,target)]
    # would silently keep the DEFAULT mass (see sweep_pendulum_alpha.py).
    env = build_pendulum_env([(0, TARGET_MASS)], [(0, DEFAULT_GRAV)])
    _hid, _nl = load_arch(args.model_dir)
    print(f'  model arch: hid={_hid} num_layers={_nl} ({_nl-1} hidden)')
    bnn, dyn = make_gaussian_bnn(OBS_DIM, ACT_DIM, n_train_layers=args.n_train_layers,
                                 hid_size=_hid, num_layers=_nl)
    dyn.load(args.model_dir)
    bnn.num_weight_groups = 1
    init_state = copy.deepcopy(bnn.state_dict())
    agent = ContinuousCEMAgent(dyn, bnn, OBS_DIM, ACT_DIM, device=device,
                               horizon=args.horizon, n_cem_iters=args.cem_iters,
                               n_candidates=args.candidates, k_models=args.k_models,
                               cvar_alpha=args.cvar_alpha, gamma=GAMMA,
                               obs_project=project_unit_circle,
                               reward_fn=pendulum_reward,
                               reward_clip=REWARD_RANGE if args.clip_reward else None)
    print(f"env check: mass={float(env.unwrapped.m):g} gravity={float(env.unwrapped.g):g}\n",
          flush=True)

    ckpt = pathlib.Path(args.model_dir) / "bnn_dynamics.pth"
    ckpt_info = dict(path=str(ckpt), md5=hashlib.md5(ckpt.read_bytes()).hexdigest()
                     ) if ckpt.exists() else {}
    config = dict(
        setting="CLAUDE.md pendulum default",
        env=dict(default_mass=DEFAULT_MASS, target_mass=TARGET_MASS,
                 gravity=DEFAULT_GRAV, change_step=CHANGE_STEP, trial_len=TRIAL_LEN),
        method=dict(forget=not args.no_forget, forget_mean=False, retrain=args.retrain,
                    retrain_variance=bool(args.retrain_variance),
                    elbo_beta=(args.elbo_beta if args.retrain_variance else None),
                    kl_denom=((args.kl_denom or "buffer") if args.retrain_variance
                              else None),
                    inflate_mode=INFLATE_MODE, q_max=Q_MAX, k_forget=args.k_forget,
                    surprise_stat=SURPRISE_STAT, surprise_clip=SURPRISE_CLIP,
                    surprise_use_reward_dim=args.surprise_reward_dim,
                    retrain_use_reward_dim=RETRAIN_USE_REWARD_DIM,
                    planner_reward="analytic_ground_truth",
                    n_train_layers=args.n_train_layers,
                    update_final_layer=bool(args.update_final_layer),
                    unfreeze_trunk=args.unfreeze_trunk,
                    adapt_scope=("B_trunk" if args.unfreeze_trunk > 0 else "A_adapter"),
                    clip_reward=bool(args.clip_reward),
                    reward_clip_range=list(REWARD_RANGE) if args.clip_reward else None),
        planner=dict(horizon=args.horizon, candidates=args.candidates,
                     cem_iters=args.cem_iters, k_models=args.k_models,
                     cvar_alpha=args.cvar_alpha, gamma_plan=GAMMA),
        model_ckpt=ckpt_info, model_dir=args.model_dir,
        device=str(device), seed_base=args.seed,
    )

    log_fh = open(log_path, "w")

    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        log_fh.write(line + "\n"); log_fh.flush()

    records, totals, t0 = [], [], time.time()
    for i in range(args.trials):
        log(f"trial {i + 1}/{args.trials}: (seed {args.seed + i})")
        rewards, n_fired, plan_stats, trace = run_trial(
            env, dyn, bnn, init_state, agent, seed=args.seed + i, args=args,
            log=None if args.no_step_log else log)
        total = float(np.sum(rewards))
        records.append(dict(seed=args.seed + i, ret=round(total, 3),
                            steps=len(rewards), forget_fired=n_fired,
                            plan_pred_return=plan_stats,
                            rewards=[round(r, 5) for r in rewards],
                            **({"trace": trace} if args.save_trace else {})))
        totals.append(total)

        # running average across FINISHED trials (what the user asked to watch)
        arr = np.array(totals)
        sem = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        elapsed = time.time() - t0
        log(f"  -> trial {i + 1:3d}/{args.trials}: return={total:8.1f}  "
            f"forget_fired={n_fired:2d}  "
            f"plan_ret[{plan_stats['min']:.1f},{plan_stats['max']:.1f}] "
            f"nonfinite={plan_stats['n_nonfinite']}  |  "
            f"RUNNING MEAN = {arr.mean():8.2f} "
            f"+/- {sem:5.2f} (sem, n={len(arr)})")
        log("  " + progress_bar(i + 1, args.trials, elapsed) + "\n")

        # incremental save so a crash never loses finished trials
        out = dict(config, n_trials_done=len(records),
                   mean_return=float(arr.mean()),
                   sem_return=float(sem), records=records)
        json_path.write_text(json.dumps(out, indent=2))

    agg = aggregate([r["rewards"] for r in records])
    write_csv(agg, csv_path)
    plot_curve(agg, png_path,
               title_suffix=f"  ·  {'forget+retrain' if args.retrain else 'forget'}")
    arr = np.array(totals)
    log(f"\n{'=' * 70}\nDONE {args.trials} trials in {fmt_time(time.time() - t0)}")
    log(f"  mean return {arr.mean():.2f} +/- "
        f"{arr.std(ddof=1) / np.sqrt(len(arr)):.2f} (sem)  "
        f"[min {arr.min():.1f}, max {arr.max():.1f}]")
    log(f"  wrote {json_path}\n         {csv_path}\n         {png_path}\n         {log_path}")
    log_fh.close()


if __name__ == "__main__":
    main()
