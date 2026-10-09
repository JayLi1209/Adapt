"""Sweep: how many PRETRAINED direction-path layers are unfrozen online.

Companion to sweep_dirichlet_layers.py, but instead of INSERTING fresh adapter
layers it gradient-updates the top `--n-unfrozen` layers of the direction path of
the SHIPPED checkpoint (Dirichlet head down into the pretrained trunk), via
bnn.unfrozen_params_dirichlet:

  n_unfrozen = 0 -> nothing trains (forget + conjugate counts only)
  n_unfrozen = 1 -> Dirichlet head only            (head-only adaptation)
  n_unfrozen = 2 -> head + top pretrained trunk layer
  n_unfrozen = 3 -> head + both pretrained trunk layers (whole path)

Scenario: [1,0,0] -> [0.7,0.15,0.15] slip change at ts=0 (model believes
deterministic).  EXPERIMENT SETTINGS for this run:
  * hole = 0  : the reported `ret` scores holes as 0 (== goal rate); the CVaR
                planner keeps its internal risk-averse hole=-1 map (settled
                convention).
  * no truncation : env TimeLimit + per-trial cap raised via --max-steps, so an
                episode ends only on goal/hole, never a step-limit timeout.
  * no discount   : gamma = 1.0.

One JSON per n_unfrozen with per-trial returns/steps/outcomes.

Run (one process per level):
    conda activate nsgym
    python sweep_unfrozen_layers.py --n-unfrozen 2 --trials 100 \
        --max-steps 1000 --out-dir results/unfrozen
"""
import argparse
import copy
import json
import pathlib
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

from config import device, SAVE_DIR, ETA, GAMMA_UNCERTAINTY
from env import build_scheduled_env
from drift import DriftFilterV2
from utils import to_one_hot_action
from bnn import (
    make_dirichlet_bnn, surprise_dirichlet, forget_dirichlet, direction_of,
    unfrozen_params_dirichlet, retrain_dirichlet, reference_entropy_table,
)
from planning.cvar_cem import CVaRCEMAgent
from planning.oracle_cem import OracleCVaRCEMAgent, true_kernel
from planning.ada_mcts import ADAMCTSAgent
from planning.mppi import MPPIAgent
from planning.ilqr import ILQRAgent
# Shared with sweep_dirichlet_layers.py so both sweeps use the SAME forget-cap and
# ELBO-retrain implementation (importing the module runs no experiment: main() is
# guarded by __main__).
from sweep_dirichlet_layers import (unfrozen_layers, variational_params,
                                    mean_weight_sigma, anchor_layers,
                                    retrain_dirichlet_elbo, predictive_variance)

SCHEDULE = [(0, 0.7)]     # slippery from step 0 (the "change"), as in run_discrete
CHANGE_STEP = 0
K_FORGET = 1
RETRAIN_LR = 1e-3
RETRAIN_STEPS = 5
N_ACTIONS = 4
GAMMA = 1.0               # NO discount (experiment convention)
ORACLE_T = None           # --oracle: the true (S,A,S) kernel CVaR-CEM plans on


def run_trials(bnn, dyn, env, init_state, n_unfrozen, n_trials, seed_base, log,
               no_forget=False, cvar_alpha=1.0, trial_len=1000,
               planner="cem", mcts_sims=3000, dpas_regular_model="prev",
               drift_window=None,
               max_forgets=None, retrain_variance=False, no_counts=False,
               step_log=None, forget_strength=1.0, rho_floor=1e-3,
               forget_trigger=1.0, log_pairs=False, symmetric_rho=False,
               lambda_cap=None, clip_delta_n=0.0, ref_entropy=False,
               ref_smooth=0.0, no_compound=False, mcts_variant="upstream",
               plan_gamma=None, retain_floor=0.0, trial_start=0,
               cem_k_models=None):
    """The run_discrete.py loop, retraining the top-`n_unfrozen` direction-path
    layers on the post-change buffer every K_FORGET steps.

    trial_len : hard per-trial step cap (safety).  With the env's own TimeLimit
    raised to match, an episode ends only on goal/hole, not truncation.
    max_forgets : cap on the number of ACTUAL forget firings (rho<1) per trial;
        deadband no-ops don't count.  2 is the measured saturation point.
    retrain_variance : retrain BOTH the posterior mean and its width via the ELBO
        (MC-sampled NLL + KL vs the prior re-anchored after each forget) instead
        of the shipped mean-weight NLL, which can only move mu.
    no_counts : switch OFF the conjugate Dirichlet-Multinomial counts.  The counts
        are themselves an adaptation channel -- they update the Dirichlet POSTERIOR
        (alpha += counts[s,a]) every step without touching any weight, and after
        forget they supply >75% of alpha.  With no_counts the model's only online
        change is forget's retain shrink, so n_unfrozen=0 becomes a genuinely
        non-learning control rather than "tabular learning, no gradient".
    step_log : open file handle.  When given, one JSON object per ENV STEP is
        written to it (JSONL), recording the full online trace: the transition
        taken, the surprise the drift filter saw, the filter state, whether
        forget/retrain fired that step and with what rho, and the resulting
        posterior scale.  Off by default -- it is ~1 line per step (up to
        trial_len * n_trials lines), so it is opt-in via --log-steps.
    """
    bnn.num_weight_groups = 20
    bnn.use_counts = not no_counts
    rng = np.random.default_rng(seed=0)
    if planner == "mcts":
        # SFIR loop unchanged; only the planner is swapped.
        agent = ADAMCTSAgent(dyn, bnn, env.env.desc, device,
                             n_actions=N_ACTIONS, rng=rng,
                             gamma=GAMMA if plan_gamma is None else plan_gamma,
                             m_simulations=mcts_sims,
                             dpas_regular_model=dpas_regular_model,
                             variant=mcts_variant)
    elif planner in ("mppi", "ilqr"):
        # Same constructor arguments as CVaR-CEM: identical models, horizon,
        # planning discount, leaf value and risk level -- only the optimiser
        # over action sequences differs.
        agent = (MPPIAgent if planner == "mppi" else ILQRAgent)(
            dyn, bnn, env.env.desc, device, n_actions=N_ACTIONS, rng=rng,
            cvar_alpha=cvar_alpha, adaptive_alpha=False,
            gamma=GAMMA if plan_gamma is None else plan_gamma)
    else:
        # cem_k_models: posterior models K per decision; the budget is
        # I*J*K*N rollouts (5*256*K*32), so K=1 gives 40960 (ADA-MCTS/RATS budget)
        cem_kw = {} if cem_k_models is None else {"k_models": cem_k_models}
        if ORACLE_T is not None:
            # --oracle: same CVaR-CEM, same I*J*K*N*H, planning on the TRUE kernel
            cem_kw["true_T"] = ORACLE_T
        agent = (OracleCVaRCEMAgent if ORACLE_T is not None else CVaRCEMAgent)(
                             dyn, bnn, env.env.desc, device,
                             n_actions=N_ACTIONS, rng=rng,
                             cvar_alpha=cvar_alpha, adaptive_alpha=False,
                             gamma=GAMMA if plan_gamma is None else plan_gamma,
                             **cem_kw)
    records = []
    # trial_start: run trials [trial_start, trial_start+n_trials) so a long run
    # can be sharded; env/torch seeds depend only on the trial index.
    for trial in range(trial_start, trial_start + n_trials):
        torch.manual_seed(seed_base + 10000 + trial)
        obs, _ = env.reset(seed=seed_base + trial)
        agent.reset()
        bnn.load_state_dict(init_state)  # pretrained weights each trial
        bnn.retain.fill_(1.0)
        bnn.reset_counts()
        train_layers = unfrozen_layers(bnn, n_unfrozen)
        retrain_params = (variational_params(train_layers) if retrain_variance
                          else unfrozen_params_dirichlet(bnn, n_unfrozen))
        retrain_opt = (torch.optim.Adam(retrain_params, lr=RETRAIN_LR)
                       if retrain_params else None)
        if retrain_variance:
            anchor_layers(train_layers)      # KL anchor = the pretrained belief
        n_forget_fired = 0
        ref_H = None
        buf_X, buf_s2 = [], []
        drift = DriftFilterV2(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY,
                              window=drift_window)
        terminated = truncated = False
        steps, post, total_return, reward = 0, 0, 0.0, 0.0
        # First moment the drift filter DETECTS the change: the first post-change
        # step whose running signal exceeds the calibrated baseline (dbar > 1,
        # i.e. lambda_hat > 0).  This is also the first step forget actually
        # fires (rho < 1) rather than sitting in the deadband.
        first_detect_step, first_detect_dbar = None, None
        t_trial = time.time()               # per-trial wall clock (planning + adaptation)
        # Visited cells s_0, s_1, ..., s_T (read-only; for occupancy maps).
        states = [int(np.argmax(obs))]
        t0 = time.time()
        while not (terminated or truncated) and steps < trial_len:
            if steps == CHANGE_STEP:
                drift.reset()
                agent.notify_change()
                if ref_entropy:
                    # Proposal 1: freeze H(p0) for every (s,a) at the change
                    # point.  Taken here -- weights just reloaded, counts reset,
                    # retain==1 -- so p0 IS the pretrained belief, and the
                    # denominator can no longer collapse as the live head
                    # sharpens on post-change counts.
                    ref_H = reference_entropy_table(dyn, bnn,
                                                    smooth=ref_smooth)
            agent.surprise_bar = drift.delta_bar
            agent.n_since_change = post
            action = agent.act(obs)
            a = int(np.argmax(action))
            one_hot_action = to_one_hot_action(action)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_return += max(0.0, float(reward))     # hole=0 scoring
            s = int(np.argmax(obs)); s2 = int(np.argmax(next_obs))
            d = direction_of(s, a, s2)
            vs = surprise_dirichlet(dyn, bnn, obs, one_hot_action, next_obs, reward,
                                    ref_entropy=ref_H)
            # Clip the raw per-step surprise BEFORE it reaches the drift filter
            # (same semantics as sweep_cw_unfrozen.py --clip-delta-n).  delta_n =
            # NLL/entropy explodes when a rare slip hits a confident deterministic
            # head (median ~95 at the first firing at p'=0.9), and the equal-weight
            # mean cannot down-weight it; clipping bounds the numerator instead.
            delta_n_raw = float(vs["delta_n"])
            delta_n = (min(delta_n_raw, clip_delta_n) if clip_delta_n > 0
                       else delta_n_raw)
            drift.update(delta_n)
            if first_detect_step is None and drift.delta_bar > 1.0:
                first_detect_step = steps
                first_detect_dbar = float(drift.delta_bar)
            if not no_counts:
                bnn.add_count(s, a, d)
            # ADA-MCTS: advance the DPAS phase counter.  Without this the
            # planner never leaves Phase 1 (worst-case sampling under M_k).
            if hasattr(agent, "observe_transition"):
                agent.observe_transition()
            if retrain_opt is not None:
                buf_X.append(np.concatenate([obs, one_hot_action]).astype(np.float32))
                buf_s2.append(s2)
            post += 1
            step_rho, step_retrained = None, False
            if post % K_FORGET == 0:
                may_forget = (not no_forget and
                              (max_forgets is None or n_forget_fired < max_forgets))
                if may_forget:
                    rho, _, _ = forget_dirichlet(bnn, drift,
                                                 strength=forget_strength,
                                                 rho_floor=rho_floor,
                                                 trigger=forget_trigger,
                                                 symmetric=symmetric_rho,
                                                 lambda_cap=lambda_cap,
                                                 compound=not no_compound)
                    step_rho = float(rho)
                    # Floor on RETAIN itself (not on rho): retain compounds
                    # over hundreds of firings, so a rho floor cannot stop it
                    # reaching ~0; this keeps >= retain_floor of the head.
                    if retain_floor > 0 and bnn.retain.item() < retain_floor:
                        bnn.retain.fill_(retain_floor)
                    if rho < 1.0:
                        n_forget_fired += 1
                        if retrain_variance:
                            # contract FROM the freshly inflated belief
                            anchor_layers(train_layers)
                if retrain_opt is not None and buf_X:
                    step_retrained = True
                    X = torch.as_tensor(np.stack(buf_X), device=device)
                    s2_idx = torch.as_tensor(
                        buf_s2, dtype=torch.long, device=device).unsqueeze(1)
                    if retrain_variance:
                        retrain_dirichlet_elbo(bnn, retrain_opt, train_layers,
                                               X, s2_idx, n_steps=RETRAIN_STEPS)
                    else:
                        retrain_dirichlet(bnn, retrain_opt, X, s2_idx,
                                          n_steps=RETRAIN_STEPS)
            if step_log is not None:
                if log_pairs:
                    _pv_mean, _pv_vec = predictive_variance(bnn, per_pair_out=True)
                else:
                    _pv_mean, _pv_vec = predictive_variance(bnn), None
                # One line per env step: the transition, what the drift filter
                # saw, and what adaptation (if any) fired in response.
                step_log.write(json.dumps(dict(
                    trial=trial, t=steps, s=s, a=a, s2=s2, direction=d,
                    reward=float(reward), terminated=bool(terminated),
                    delta_n=float(delta_n),
                    delta_n_raw=float(delta_n_raw),
                    delta_bar=float(drift.delta_bar),
                    lambda_hat=float(drift.lambda_hat),
                    detected=bool(drift.delta_bar > 1.0),
                    alpha_cvar=float(getattr(agent, "last_alpha", 0.0)),
                    rho=step_rho, forget_fired=bool(step_rho is not None
                                                    and step_rho < 1.0),
                    n_forget_fired=n_forget_fired,
                    retrained=step_retrained,
                    retain=float(bnn.retain.item()),
                    # Mean total Dirichlet variance sum_i Var(p_i) over all
                    # (s,a), read AFTER this step's counts / forget / retrain --
                    # i.e. the belief width the NEXT decision is made with.
                    # Pure read-out: it touches no state, so the trajectory is
                    # unchanged whether or not --log-steps is on.
                    pvar=round(_pv_mean, 8),
                    # Per-(s,a) variance vector (64 entries, s*4+a order) and the
                    # pair visited THIS step, so a plot can restrict the average
                    # to the pairs the agent actually uses.  The 64-pair mean is
                    # dominated by the ~53 pairs that never get a sample and sit
                    # at the post-forget ceiling forever.
                    pvar_pairs=([round(float(v), 6) for v in _pv_vec]
                                if log_pairs else None),
                    pair=s * N_ACTIONS + a,
                )) + "\n")
            obs = next_obs
            states.append(s2)
            steps += 1
        outcome = ("goal" if reward > 0 else "hole") if terminated else "truncated"
        records.append(dict(trial=trial, ret=total_return, steps=steps, outcome=outcome,
                            secs=time.time() - t_trial,
                            first_detect_step=first_detect_step,
                            first_detect_dbar=first_detect_dbar,
                            final_dbar=float(drift.delta_bar),
                            final_retain=float(bnn.retain.item()),
                            n_forget_fired=n_forget_fired,
                            sigma_final=round(mean_weight_sigma(train_layers), 8),
                            states=states))
        _fd = ("t=%d dbar=%.4f" % (first_detect_step, first_detect_dbar)
               if first_detect_step is not None else "never")
        log(f"  n_unfrozen={n_unfrozen} trial {trial+1}/{n_trials}: "
            f"{outcome} steps={steps} ret={total_return:.0f} "
            f"| first_detect[{_fd}] | final dbar={drift.delta_bar:.4f} "
            f"retain={float(bnn.retain.item()):.3e} | forget fired {n_forget_fired}"
            f" | sigma={mean_weight_sigma(train_layers):.6f} ({time.time()-t0:.0f}s)")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-unfrozen", type=int, required=True,
                    help="top direction-path layers to gradient-update online "
                         "(0=frozen, 1=head, 2=head+trunk1, 3=head+trunk1+trunk2)")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--trial-start", type=int, default=0,
                    help="first trial index (for sharding a long run)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/unfrozen")
    ap.add_argument("--no-forget", action="store_true")
    ap.add_argument("--cvar-alpha", type=float, default=1.0)
    ap.add_argument("--planner", default="cem", choices=("cem", "mcts", "mppi", "ilqr"),
                    help="cem = CVaR-CEM (default); mcts = ADA-MCTS. The SFIR "
                         "adaptation loop is identical either way.")
    ap.add_argument("--mcts-sims", type=int, default=3000,
                    help="MCTS simulations per action (only with --planner mcts).")
    ap.add_argument("--mcts-variant", default="upstream",
                    choices=("upstream", "alg2"),
                    help="upstream = port of ADA-MCTS/adamcts.py; alg2 = the "
                         "paper's Algorithm 2 line by line (see ada_mcts.py)")
    ap.add_argument("--retain-floor", type=float, default=0.0, metavar="R",
                    help="clamp retain >= R after every forget (0 = off). "
                         "Unlike --rho-floor this bounds the COMPOUNDED retain")
    ap.add_argument("--plan-gamma", type=float, default=None,
                    help="PLANNING-only discount for the planner (the env "
                         "and the reported return stay undiscounted). Default: "
                         "GAMMA (1.0)")
    ap.add_argument("--drift-window", type=int, default=None, metavar="K",
                    help="DriftFilterV2 window: average lambda_hat over only the "
                         "LAST K post-change samples instead of all of them "
                         "(default None = cumulative).  A cumulative filter lets "
                         "the first huge delta_n latch delta_bar high forever, so "
                         "forgetting keeps firing long after the belief is fixed.")
    ap.add_argument("--dpas-regular-model", default="prev", choices=("prev", "k"),
                    help="Which model supplies DPAS's regular Phase-2 sample. "
                         "'prev' = M_(k-1), reproducing upstream adamcts.py; "
                         "'k' = M_k, which is what Algorithm 2 line 56 specifies.")
    ap.add_argument("--oracle", action="store_true",
                    help="CVaR-CEM plans on the TRUE post-change kernel instead of "
                         "the BNN (same I*J*K*N*H budget); planning/oracle_cem.py")
    ap.add_argument("--cem-k-models", type=int, default=None, metavar="K",
                    help="CVaR-CEM posterior models per decision (default: "
                         "planner's K_MODELS=10). Rollouts/decision = 5*256*K*32.")
    ap.add_argument("--max-steps", type=int, default=1000,
                    help="env TimeLimit AND per-trial cap; large => no truncation")
    ap.add_argument("--max-forgets", type=int, default=None, metavar="K",
                    help="cap the number of ACTUAL forget firings (rho<1) per "
                         "trial; deadband no-ops don't count (default: no cap)")
    ap.add_argument("--no-counts", action="store_true",
                    help="switch off the conjugate count updates, so the model's "
                         "Dirichlet posterior is NOT updated online (a true "
                         "no-learning control at --n-unfrozen 0)")
    ap.add_argument("--retrain-variance", action="store_true",
                    help="ELBO retrain of BOTH mu and rho (posterior mean AND "
                         "width); without it only mu can move")
    ap.add_argument("--log-steps", action="store_true",
                    help="also write a PER-STEP JSONL trace (<tag>_steps.jsonl): "
                         "one line per env step with the transition, surprise, "
                         "drift state, and the forget/retrain that fired")
    ap.add_argument("--ckpt", default=None, metavar="PATH",
                    help="path to the pretrained checkpoint to start from "
                         "(default: the shipped DETERMINISTIC FrozenLake model). "
                         "Use e.g. data/frozenlake_p07/bnn_dirichlet_frozenlake"
                         "_k3_p0p7.pth to start from a p=0.7 belief instead")
    ap.add_argument("--tag-suffix", default="", metavar="S",
                    help="extra string appended to the output tag, to keep runs "
                         "with different --ckpt from colliding")
    ap.add_argument("--forget-strength", type=float, default=1.0, metavar="S",
                    help="scale HOW HARD each forget firing pulls: "
                         "rho_eff = 1 - S*(1-rho).  1.0 = shipped, 0.5 = half "
                         "shrink per firing, 0.0 = forgetting off")
    ap.add_argument("--rho-floor", type=float, default=1e-3, metavar="F",
                    help="lower clip on rho (default 1e-3, the shipped value). "
                         "Because retain COMPOUNDS, this bounds the asymptote: "
                         "a floor of 0.99 keeps retain ~0.13 after 200 firings, "
                         "the shipped 1e-3 sends it to 0")
    ap.add_argument("--log-pairs", action="store_true",
                    help="with --log-steps, also record pvar_pairs (the 64 "
                         "per-(s,a) variances) and the pair visited each step, "
                         "so a plot can average over only the VISITED pairs")
    ap.add_argument("--forget-trigger", type=float, default=1.0, metavar="T",
                    help="delta_bar must EXCEED T before forgetting fires "
                         "(default 1.0 = shipped: fire whenever above the "
                         "calibrated baseline, ~260x/trial at p'=0.9). "
                         "Raising it (e.g. 10) demands a much larger filtered "
                         "surprise before any belief is discarded; the pull "
                         "magnitude above the trigger is unchanged")
    ap.add_argument("--symmetric-rho", action="store_true",
                    help="drive rho by |lambda_hat| instead of the signed "
                         "delta_bar: rho = 1/(1+|lambda_hat|), gated on the "
                         "magnitude.  Makes forgetting fire on a change TOWARD "
                         "determinism (lambda_hat < 0), which the shipped path "
                         "discards at BOTH the trigger gate and the "
                         "max(delta_bar,1) clamp")
    ap.add_argument("--change-p", type=float, default=None, metavar="P",
                    help="post-change intended-direction probability at ts=0 "
                         "(default: the module SCHEDULE, p=0.7).  The model is "
                         "always pretrained deterministic, so this sets the SIZE "
                         "of the change: [1,0,0] -> [P, (1-P)/2, (1-P)/2]")
    ap.add_argument("--no-compound", action="store_true",
                    help="retain <- rho instead of retain <- retain*rho, so the "
                         "retained fraction reflects the CURRENT drift rather "
                         "than the product of every pull since the change.  "
                         "Stops retain being an absorbing state at 0 and lets it "
                         "RECOVER when the filter calms down")
    ap.add_argument("--ref-entropy", action="store_true",
                    help="PROPOSAL 1: normalise the surprise by the entropy of a "
                         "FIXED reference predictive H(p0) captured at the change "
                         "point, instead of the LIVE H(p_t).  Keeps E[delta_n]=1 "
                         "under the null (p_t==p0) while bounding the spike to "
                         "-log p(s2)/H(p0) -- logarithmic rather than 1/H")
    ap.add_argument("--ref-smooth", type=float, default=0.0, metavar="EPS",
                    help="with --ref-entropy, mix the reference predictive "
                         "toward uniform by EPS before taking its entropy: "
                         "p0_eps = (1-EPS)*p0 + EPS/k_dir.  Needed when the "
                         "snapshot is a (near-)deterministic pretrained belief, "
                         "whose raw H(p0) ~ 0 would inflate every delta_n")
    ap.add_argument("--clip-delta-n", type=float, default=0.0, metavar="C",
                    help="clip the raw per-step surprise delta_n at C before it "
                         "reaches the drift filter (0 = no clip, the shipped "
                         "behaviour).  Bounds the NUMERATOR of lambda_hat, which "
                         "--rho-floor / --lambda-cap cannot do: they clip the "
                         "CONSEQUENCE of a spike, this clips the spike itself")
    ap.add_argument("--lambda-cap", type=float, default=None, metavar="L",
                    help="ceiling on the drift estimate that drives rho: "
                         "rho = 1/(1+min(lambda_hat, L)), so no single firing "
                         "pulls harder than 1/(1+L).  Same arithmetic bound as "
                         "--rho-floor 1/(1+L) but expressed in the logged "
                         "lambda_hat units (default: uncapped, as shipped)")
    args = ap.parse_args()

    global SCHEDULE, ORACLE_T
    if args.change_p is not None:
        SCHEDULE = [(0, args.change_p)]
    if args.oracle:
        if args.planner != "cem" or len(SCHEDULE) != 1 or SCHEDULE[0][0] != 0:
            raise SystemExit("--oracle needs --planner cem and a single change at ts 0")
        from grids import get_grid
        ORACLE_T = true_kernel(get_grid("frozenlake"), SCHEDULE[0][1])

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (("oracle_" if args.oracle else "") + f"U{args.n_unfrozen}"
           + ("_nf" if args.no_forget else "")
           + (f"_p{args.change_p:g}".replace(".", "p")
              if args.change_p is not None else "")
           + (f"_s{args.forget_strength:g}".replace(".", "p")
              if args.forget_strength != 1.0 else "")
           + (f"_rf{args.rho_floor:g}".replace(".", "p").replace("-", "m")
              if args.rho_floor != 1e-3 else "")
           + (f"_tg{args.forget_trigger:g}".replace(".", "p")
              if args.forget_trigger != 1.0 else "")
           + (f"_lc{args.lambda_cap:g}".replace(".", "p")
              if args.lambda_cap is not None else "")
           + (f"_clip{args.clip_delta_n:g}".replace(".", "p")
              if args.clip_delta_n > 0 else "")
           + ("_nocomp" if args.no_compound else "")
           + ("_refH" if args.ref_entropy else "")
           + (f"_sm{args.ref_smooth:g}".replace(".", "p")
              if args.ref_entropy and args.ref_smooth > 0 else "")
           + (f"_f{args.max_forgets}" if args.max_forgets is not None else "")
           + ("_rv" if args.retrain_variance else "")
           + ("_nc" if args.no_counts else "")
           + ("_symrho" if args.symmetric_rho else "")
           + (f"_rtf{args.retain_floor:g}".replace(".", "p")
              if args.retain_floor > 0 else "")
           + ("_alg2" if args.mcts_variant == "alg2" else "")
           + (f"_{args.planner}" if args.planner in ("mppi", "ilqr") else "")
           + (f"_t{args.trial_start}" if args.trial_start else "")
           + args.tag_suffix)

    def log(*a):
        print(f"[{tag}]", *a, flush=True)

    torch.manual_seed(args.seed + args.n_unfrozen)
    np.random.seed(args.seed + args.n_unfrozen)

    env = build_scheduled_env(SCHEDULE, max_episode_steps=args.max_steps)
    bnn, dyn = make_dirichlet_bnn(16, 4)
    if args.ckpt is not None:
        # A checkpoint pretrained on some OTHER original-p (e.g. the p=0.7 model
        # from pretrain_gridworld.py).  Changes what "the model believes before
        # the change", so --change-p is then a change RELATIVE to that belief.
        ck = pathlib.Path(args.ckpt)
        bnn.load(ck.parent, filename=ck.name)
    else:
        bnn.load(SAVE_DIR)              # shared shipped checkpoint, all configs
    init_state = copy.deepcopy(bnn.state_dict())
    log(f"loaded shipped checkpoint; n_unfrozen={args.n_unfrozen} "
        f"trial_len={args.max_steps} gamma={GAMMA} hole=0-scored")

    step_path = out_dir / f"{tag}_steps.jsonl" if args.log_steps else None
    # buffering=1 (line-buffered): the trace is readable WHILE the run is in
    # flight, so a long sweep can be checked without waiting for the close.
    step_log = open(step_path, "w", buffering=1) if step_path else None
    if step_log:
        log(f"per-step trace -> {step_path}")
    try:
        records = run_trials(bnn, dyn, env, init_state, args.n_unfrozen,
                             args.trials, seed_base=args.seed + 1000,
                             log=log, no_forget=args.no_forget,
                             cvar_alpha=args.cvar_alpha, trial_len=args.max_steps,
                             planner=args.planner, mcts_sims=args.mcts_sims,
                             dpas_regular_model=args.dpas_regular_model,
                             drift_window=args.drift_window,
                             max_forgets=args.max_forgets,
                             retrain_variance=args.retrain_variance,
                             no_counts=args.no_counts, step_log=step_log,
                             symmetric_rho=args.symmetric_rho,
                             forget_strength=args.forget_strength,
                             rho_floor=args.rho_floor,
                             forget_trigger=args.forget_trigger,
                             lambda_cap=args.lambda_cap,
                             clip_delta_n=args.clip_delta_n,
                             ref_entropy=args.ref_entropy,
                             ref_smooth=args.ref_smooth,
                             no_compound=args.no_compound,
                             mcts_variant=args.mcts_variant,
                             plan_gamma=args.plan_gamma,
                             retain_floor=args.retain_floor,
                             trial_start=args.trial_start,
                             cem_k_models=args.cem_k_models,
                             log_pairs=args.log_pairs)
    finally:
        if step_log:
            step_log.close()
    rets = [r["ret"] for r in records]
    goals = [r["outcome"] == "goal" for r in records]
    # ── first-detection stats: dbar at the FIRST step the filter saw the change ──
    fd_dbar = [r["first_detect_dbar"] for r in records
               if r["first_detect_dbar"] is not None]
    fd_step = [r["first_detect_step"] for r in records
               if r["first_detect_step"] is not None]
    fd_n = len(fd_dbar)
    fd_dbar_mean = float(np.mean(fd_dbar)) if fd_n else float("nan")
    fd_dbar_sem = (float(np.std(fd_dbar, ddof=1) / np.sqrt(fd_n))
                   if fd_n > 1 else 0.0)
    fd_step_mean = float(np.mean(fd_step)) if fd_n else float("nan")
    out = dict(
        n_unfrozen=args.n_unfrozen, forget=not args.no_forget, oracle=args.oracle,
        cvar_alpha=args.cvar_alpha, gamma=GAMMA, planner=args.planner,
        mcts_variant=args.mcts_variant, plan_gamma=args.plan_gamma,
        mcts_sims=args.mcts_sims, cem_k_models=args.cem_k_models,
        retain_floor=args.retain_floor, drift_window=args.drift_window, trials=args.trials,
        seed=args.seed, schedule=SCHEDULE, trial_len=args.max_steps,
        k_forget=K_FORGET, max_forgets=args.max_forgets, counts=not args.no_counts,
        symmetric_rho=args.symmetric_rho,
        forget_strength=args.forget_strength, rho_floor=args.rho_floor,
        forget_trigger=args.forget_trigger, lambda_cap=args.lambda_cap,
        clip_delta_n=args.clip_delta_n, ref_entropy=args.ref_entropy,
        ref_smooth=args.ref_smooth, no_compound=args.no_compound,
        ckpt=(args.ckpt if args.ckpt is not None else str(SAVE_DIR)),
        retrain_variance=args.retrain_variance,
        retrain_lr=RETRAIN_LR, retrain_steps=RETRAIN_STEPS,
        hole_scoring=0, mean_return=float(np.mean(rets)),
        sem_return=float(np.std(rets, ddof=1) / np.sqrt(len(rets))),
        goal_rate=float(np.mean(goals)),
        trunc_rate=float(np.mean([r["outcome"] == "truncated" for r in records])),
        first_detect_dbar_mean=fd_dbar_mean,
        first_detect_dbar_sem=fd_dbar_sem,
        first_detect_step_mean=fd_step_mean,
        first_detect_n_trials=fd_n,
        records=records,
    )
    path = out_dir / f"{tag}.json"
    path.write_text(json.dumps(out, indent=2))
    log(f"saved {path}: mean return {np.mean(rets):.3f} "
        f"goal rate {np.mean(goals):.3f} "
        f"trunc {out['trunc_rate']:.3f}")
    log(f"  FIRST-DETECTION (dbar>1): mean dbar={fd_dbar_mean:.4f} "
        f"+/- {fd_dbar_sem:.4f} (SEM) at mean step={fd_step_mean:.2f} "
        f"| detected in {fd_n}/{args.trials} trials")


if __name__ == "__main__":
    main()
