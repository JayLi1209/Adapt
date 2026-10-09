"""ns-Bridge + gradient retraining: the sweep_unfrozen_layers.py mechanism
(the FrozenLake run that scores ~0.83), ported to the 5x8 bridge grid.

Identical adaptation stack to the FrozenLake sweep and to sweep_cw_unfrozen.py:
surprise -> drift -> forget(inflate) + conjugate counts + gradient retrain of the
top `--n-unfrozen` layers of the Dirichlet direction path.

  n_unfrozen = 0 -> nothing trains (forget + counts only)
  n_unfrozen = 1 -> Dirichlet head only
  n_unfrozen = 2 -> head + top pretrained trunk layer
  n_unfrozen = 3 -> head + both pretrained trunk layers

Retrain mechanism (identical to the FrozenLake sweep):
  * Adam, lr=1e-3, 5 inner steps, fired every K_FORGET=1 env step
  * loss = categorical NLL of the realized cell under the MEAN-weight forward
  * buffer = every post-change (obs,act) -> s2 seen this trial, replayed whole
  * only mu is updated; rho stays fixed so surprise/forget keep their meaning

Bridge specifics (vs FrozenLake / CliffWalking):
  * K = 3, slip_mode="perp": with prob p the intended direction, each
    PERPENDICULAR direction (1-p)/2 -- the same slip structure as FrozenLake,
    and what makes the narrow right arm of the bridge genuinely lethal.
  * Action order is FrozenLake's: LEFT, DOWN, RIGHT, UP.
  * grid.direction_of / grid.desc_bytes(), as in sweep_cw_unfrozen.py -- the bnn
    helpers are hardcoded to FrozenLake 4x4 geometry.

Scenario for this run: pretrained p=1.0 (deterministic prior) -> p'=0.7 at ts=0,
gamma=1.0 (no discount), no truncation, holes scored 0 in the reported return.

    python sweep_bridge_unfrozen.py --n-unfrozen 1 --change-p 0.7 --trials 100
"""
import argparse
import copy
import json
import pathlib
import time
import warnings

import os
import sys

# This script lives in Bridge/; the shared modules (config, grids, bnn, env,
# planning, utils, drift) live at the repo root, so put the root on sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# sweep_dirichlet_layers (adapter helpers shared with FrozenLake) lives in FrozenLake/.
_FL_DIR = os.path.join(_REPO_ROOT, "FrozenLake")
if _FL_DIR not in sys.path:
    sys.path.insert(1, _FL_DIR)

# Bridge checkpoints live under Bridge/data/<grid>/, not the repo-root data/.
BRIDGE_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

from config import device, SAVE_DIR, ETA, GAMMA_UNCERTAINTY
from env.gridworlds import build_env
from grids import get_grid
from drift import DriftFilterV2
from utils import to_one_hot_action
from bnn import (make_dirichlet_bnn, surprise_dirichlet, forget_dirichlet,
                 unfrozen_params_dirichlet, retrain_dirichlet)
from bnn.dirichlet_model import ckpt_name
from planning.cvar_cem import CVaRCEMAgent
from planning.oracle_cem import OracleCVaRCEMAgent, true_kernel
from planning.ada_mcts import ADAMCTSAgent
from sweep_dirichlet_layers import unfrozen_layers, mean_weight_sigma

CHANGE_STEP = 0
K_FORGET = 1
RETRAIN_LR = 1e-3         # same as sweep_unfrozen_layers.py
RETRAIN_STEPS = 5         # same as sweep_unfrozen_layers.py (--retrain-steps overrides)
GAMMA = 1.0               # NO discount in the reported SCORE (see --plan-gamma)
ORIGINAL_P = 1.0          # deterministic prior == CliffWalking "Sweep B"

# Raw per-step surprise (delta_n = -log p(s2)/H(p)) runs ~100x hotter on the 4x12
# grid than on FrozenLake 4x4: measured under a matched deterministic-prior loop,
# FrozenLake's delta_n has p99 = 42.6 / max = 60.9, while CliffWalking reaches
# p99 = 3464 / max = 3992.  Because forget uses rho = 1/max(delta_bar,1), that
# tail slams retain into the 1e-3 clamp floor within a few steps and the
# gradient-retrain channel is algebraically multiplied out (see
# dirichlet_model.py: alpha = CONC_PRIOR + retain*(alpha_head - CONC_PRIOR)).
# Clipping delta_n at FrozenLake's observed ceiling puts the two grids' drift
# signals on the same scale.
DELTA_N_CLIP_FL = 50.0    # ~FrozenLake p99 (42.6); clips 2.3% of FL steps

ACT_NAMES = ("LEFT", "DOWN", "RIGHT", "UP")   # grids.BRIDGE_5x8 delta order
DIR_NAMES = ("intended", "perp-", "perp+")    # K=3, slip_mode="perp"


def run_trials(args, grid, bnn, dyn, env, init_state, log, trace_fh):
    bnn.num_weight_groups = 20
    bnn.use_counts = not args.no_counts
    rng = np.random.default_rng(seed=0)
    kw = {} if args.horizon is None else {"horizon": args.horizon}
    kw["pi_kappa"] = args.pi_kappa
    # PLANNING discount is decoupled from the reported score: the env return is
    # always summed undiscounted (GAMMA=1.0, hole=0-scored), while the planner
    # imagines with args.plan_gamma.  This matters on the bridge because the leaf
    # heuristic is V_h(s) = plan_gamma^dist(s,goal): at plan_gamma=1.0 it is flat
    # 1.0 everywhere, so beyond the H=6 horizon the planner has NO gradient toward
    # the goal and dithers.
    if args.planner == "mcts":
        # ADA-MCTS shares BNNModelPlanner and the act(obs) interface, so it drops
        # into the same loop.  It has no CVaR tail / step-cost knobs of its own.
        agent = ADAMCTSAgent(dyn, bnn, grid.desc_bytes(), device,
                             n_actions=grid.n_actions, rng=rng,
                             gamma=args.plan_gamma,
                             m_simulations=args.mcts_sims,
                             h_rollout=(args.horizon or 6))
    else:
        if args.oracle:
            # --oracle: same CVaR-CEM, same I*J*K*N*H, planning on the TRUE kernel
            kw = dict(kw, true_T=true_kernel(grid, args.change_p))
        agent = (OracleCVaRCEMAgent if args.oracle else CVaRCEMAgent)(
                             dyn, bnn, grid.desc_bytes(), device,
                             n_actions=grid.n_actions, rng=rng,
                             cvar_alpha=args.cvar_alpha,
                             adaptive_alpha=args.adaptive_alpha,
                             gamma=args.plan_gamma, step_cost=args.step_cost, **kw)
    records = []
    for trial in range(args.trials):
        torch.manual_seed(args.seed + 10000 + trial)
        obs, _ = env.reset(seed=args.seed + 1000 + trial)
        agent.reset()
        bnn.load_state_dict(init_state)
        bnn.retain.fill_(1.0)
        bnn.reset_counts()
        if args.conc_strength is not None:   # after load_state_dict: it restores buffers
            bnn.conc_prior.fill_(float(args.conc_strength))
        if args.conc_prior is not None:
            bnn.set_conc_prior(args.conc_prior)
        bnn.anchor_forget = args.anchor_forget
        bnn.reset_alpha_state()

        train_layers = unfrozen_layers(bnn, args.n_unfrozen)
        retrain_params = unfrozen_params_dirichlet(bnn, args.n_unfrozen)
        retrain_opt = (torch.optim.Adam(retrain_params, lr=RETRAIN_LR)
                       if retrain_params else None)

        n_forget_fired = 0
        trig_rng = np.random.default_rng(args.seed + 777 + trial)
        buf_X, buf_s2 = [], []
        drift = DriftFilterV2(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)
        terminated = truncated = False
        steps, post, total_return, reward = 0, 0, 0.0, 0.0
        first_detect_step, first_detect_dbar = None, None
        t0 = time.time()
        while not (terminated or truncated) and steps < args.max_steps:
            if steps == CHANGE_STEP:
                drift.reset()
                agent.notify_change()
            agent.surprise_bar = drift.delta_bar
            agent.n_since_change = post
            action = agent.act(obs)
            a = int(np.argmax(action))
            one_hot_action = to_one_hot_action(action)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_return += max(0.0, float(reward))       # hole=0 scoring
            s = int(np.argmax(obs)); s2 = int(np.argmax(next_obs))
            d = grid.direction_of(s, a, s2)               # GRID method, not bnn's
            # In const-rho / random-trigger mode the surprise statistic is not
            # used at all: delta_n, lambda_hat and kappa are removed from the loop.
            if args.const_rho is None:
                vs = surprise_dirichlet(dyn, bnn, obs, one_hot_action, next_obs, reward)
                delta_n_raw = float(vs["delta_n"])
                delta_n = (min(delta_n_raw, args.clip_delta_n) if args.clip_delta_n > 0
                           else delta_n_raw)
                drift.update(delta_n)
            else:
                delta_n_raw = delta_n = float("nan")
            if args.const_rho is None and first_detect_step is None and drift.delta_bar > 1.0:
                first_detect_step = steps
                first_detect_dbar = float(drift.delta_bar)
            if not args.no_counts:
                bnn.add_count(s, a, d)
            if retrain_opt is not None:
                buf_X.append(np.concatenate([obs, one_hot_action]).astype(np.float32))
                buf_s2.append(s2)

            post += 1
            rho = 1.0
            nll = None
            if post % K_FORGET == 0:
                may_forget = (not args.no_forget and
                              (args.max_forgets is None
                               or n_forget_fired < args.max_forgets))
                if may_forget and args.const_rho is not None:
                    # ABLATION: fixed rho, no surprise statistic anywhere.
                    # --trigger-rate p fires only with probability p (matched-rate
                    # random trigger), isolating WHEN it fires from HOW OFTEN.
                    fire = (args.trigger_rate is None
                            or trig_rng.random() < args.trigger_rate)
                    if fire:
                        rho = float(args.const_rho)
                        if getattr(bnn, "anchor_forget", False):
                            bnn.anchor_pull(rho)
                        else:
                            bnn.retain.fill_(float(bnn.retain.item()) * rho)
                        n_forget_fired += 1
                elif may_forget:
                    rho, _, _ = forget_dirichlet(bnn, drift)
                    if rho < 1.0:          # deadband no-ops don't count toward the cap
                        n_forget_fired += 1
                if retrain_opt is not None and buf_X:
                    X = torch.as_tensor(np.stack(buf_X), device=device)
                    s2_idx = torch.as_tensor(buf_s2, dtype=torch.long,
                                             device=device).unsqueeze(1)
                    nll = retrain_dirichlet(bnn, retrain_opt, X, s2_idx,
                                            n_steps=args.retrain_steps)

            if trace_fh is not None:
                trace_fh.write(json.dumps(dict(
                    trial=trial, step=steps, s=s, a=a, act=ACT_NAMES[a], s2=s2,
                    dir=d, dir_name=(DIR_NAMES[d] if d >= 0 else "unexplained"),
                    reward=float(reward), ret_so_far=total_return,
                    delta_n=delta_n, delta_n_raw=delta_n_raw,
                    clipped=bool(delta_n_raw > delta_n),
                    dbar=float(drift.delta_bar),
                    rho=float(rho), retain=float(bnn.retain.item()),
                    retrain_nll=nll, buf=len(buf_X),
                    sigma=round(mean_weight_sigma(train_layers), 8),
                    margin=float(getattr(agent, "last_margin", float("nan"))),
                    terminated=bool(terminated), truncated=bool(truncated),
                )) + "\n")
                trace_fh.flush()
            obs = next_obs
            steps += 1

        outcome = ("goal" if reward > 0 else "hole") if terminated else "truncated"
        records.append(dict(ret=total_return, steps=steps, outcome=outcome,
                            secs=time.time() - t0,
                            first_detect_step=first_detect_step,
                            first_detect_dbar=first_detect_dbar,
                            final_dbar=float(drift.delta_bar),
                            final_retain=float(bnn.retain.item()),
                            n_forget_fired=n_forget_fired,
                            sigma_final=round(mean_weight_sigma(train_layers), 8)))
        _fd = ("t=%d dbar=%.4f" % (first_detect_step, first_detect_dbar)
               if first_detect_step is not None else "never")
        log(f"  trial {trial+1}/{args.trials}: {outcome:9s} steps={steps:4d} "
            f"ret={total_return:.0f} | first_detect[{_fd}] "
            f"| final dbar={drift.delta_bar:.4f} retain={float(bnn.retain.item()):.3e} "
            f"| forget fired {n_forget_fired} | sigma={mean_weight_sigma(train_layers):.6f} "
            f"({time.time()-t0:.0f}s)")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-unfrozen", type=int, required=True)
    ap.add_argument("--change-p", type=float, required=True)
    ap.add_argument("--original-p", type=float, default=ORIGINAL_P)
    ap.add_argument("--cvar-alpha", type=float, default=0.0)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--no-forget", action="store_true")
    ap.add_argument("--no-counts", action="store_true",
                    help="switch OFF the conjugate Dirichlet-Multinomial counts. "
                         "With --no-forget and --n-unfrozen 0 this is the "
                         "genuine no-adaptation control.")
    ap.add_argument("--clip-delta-n", type=float, default=0.0,
                    help="clip the raw per-step surprise delta_n at this value "
                         "before it reaches the drift filter (0 = no clip). "
                         f"{DELTA_N_CLIP_FL:g} matches FrozenLake's observed ceiling.")
    ap.add_argument("--const-rho", type=float, default=None,
                    help="ABLATION: apply this FIXED rho every forget step and "
                         "remove the surprise statistic (delta_n / lambda_hat / "
                         "kappa) from the loop entirely.")
    ap.add_argument("--trigger-rate", type=float, default=None,
                    help="with --const-rho, fire only with this probability per "
                         "step (matched-rate random trigger), separating WHEN a "
                         "forget fires from HOW OFTEN.")
    ap.add_argument("--pi-kappa", type=float, default=0.0,
                    help="CEM re-widening: pi <- (1-kappa)*greedy + kappa*uniform "
                         "on the proposal seed (Algorithm 3 line 4 analogue).")
    ap.add_argument("--anchor-forget", action="store_true",
                    help="Algorithm 3 line 26 semantics: forget applies the affine "
                         "pull alpha <- alpha_0 + rho*(alpha - alpha_0) to a "
                         "persistent per-(s,a) alpha, making --conc-prior a true "
                         "anchor instead of a starting point the shrink discards.")
    ap.add_argument("--max-forgets", type=int, default=None, metavar="K",
                    help="cap the number of ACTUAL forget firings (rho<1) per "
                         "trial; deadband no-ops don't count.  Keeps retain from "
                         "collapsing, so the gradient-retrain channel is not "
                         "scaled out of the prediction.")
    ap.add_argument("--conc-strength", type=float, default=None, metavar="C",
                    help="SCALAR total-strength knob: set every conc_prior entry "
                         "to C (module default CONC_PRIOR=0.1).  This is the "
                         "STRENGTH of the forgotten belief -- alpha0 = K*C -- as "
                         "opposed to --conc-prior, which only reshapes at fixed "
                         "strength.  Smaller C = a weaker prior, so the conjugate "
                         "counts dominate sooner after a forget.")
    ap.add_argument("--conc-prior", type=float, nargs=3, default=None,
                    metavar=("INTENDED", "PERP_MINUS", "PERP_PLUS"),
                    help="relative per-direction prior weights over "
                         "[intended, perp-, perp+]; rescaled to the symmetric "
                         "prior's total strength, so [1,1,1] == default.")
    ap.add_argument("--horizon", type=int, default=None,
                    help="CEM planning horizon H (default: planner's H_PLAN=6)")
    ap.add_argument("--plan-gamma", type=float, default=1.0,
                    help="discount used INSIDE the planner (rollout return and "
                         "the leaf heuristic gamma^dist).  The reported score is "
                         "always undiscounted.  1.0 reproduces the old runs; "
                         "0.95 is the planner's own default (PLAN_GAMMA).")
    ap.add_argument("--step-cost", type=float, default=0.0,
                    help="per-step living cost charged inside the planner only "
                         "(env reward untouched, so goal rate keeps its meaning).")
    ap.add_argument("--adaptive-alpha", action="store_true",
                    help="let the CVaR alpha adapt to surprise/confidence rather "
                         "than staying pinned at --cvar-alpha.")
    ap.add_argument("--retrain-steps", type=int, default=RETRAIN_STEPS,
                    help="inner gradient steps per retrain firing (0.83 setup: 5).")
    ap.add_argument("--oracle", action="store_true",
                    help="CVaR-CEM plans on the TRUE post-change kernel instead of "
                         "the BNN (same I*J*K*N*H budget); planning/oracle_cem.py")
    ap.add_argument("--planner", default="cem", choices=("cem", "mcts"),
                    help="cem = CVaR-CEM (default); mcts = ADA-MCTS (planning/ada_mcts.py).")
    ap.add_argument("--mcts-sims", type=int, default=3000,
                    help="MCTS simulations per action (only with --planner mcts).")
    ap.add_argument("--trace", action="store_true", help="write per-step JSONL")
    ap.add_argument("--out-dir", default="results/bridge_unfrozen")
    ap.add_argument("--grid", default="bridge", choices=("bridge", "bridge_h17", "bridge_h18", "bridge_h19"),
                    help="bridge = 17,18 both FREE; bridge_h17 = cell 17 HOLE; "
                         "bridge_h18 = cell 18 HOLE with 17 FREE.")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = (("oracle_" if args.oracle else "") + f"U{args.n_unfrozen}_p{args.change_p:g}_a{args.cvar_alpha:g}"
           + (f"_clip{args.clip_delta_n:g}" if args.clip_delta_n > 0 else "")
           + (f"_H{args.horizon}" if args.horizon is not None else "")
           + (f"_f{args.max_forgets}" if args.max_forgets is not None else "")
           + ("_anch" if args.anchor_forget else "")
           + (f"_cr{args.const_rho:g}" if args.const_rho is not None else "")
           + (f"_tr{args.trigger_rate:g}" if args.trigger_rate is not None else "")
           + (f"_k{args.pi_kappa:g}" if args.pi_kappa > 0 else "")
           + (f"_pg{args.plan_gamma:g}" if args.plan_gamma != 1.0 else "")
           + (f"_sc{args.step_cost:g}" if args.step_cost > 0 else "")
           + ("_aa" if args.adaptive_alpha else "")
           + (f"_mcts{args.mcts_sims}" if args.planner == "mcts" else "")
           + (f"_rs{args.retrain_steps}" if args.retrain_steps != RETRAIN_STEPS else "")
           + ("_nf" if args.no_forget else "")
           + ("_nc" if args.no_counts else "")
           + ("_cp" + "".join(f"{w:g}" for w in args.conc_prior)
              if args.conc_prior is not None else "")
           + (f"_cs{args.conc_strength:g}" if args.conc_strength is not None else "")
           ).replace(".", "p")

    def log(*a):
        print(f"[{tag}]", *a, flush=True)

    torch.manual_seed(args.seed + args.n_unfrozen)
    np.random.seed(args.seed + args.n_unfrozen)

    env, grid = build_env(args.grid, [(0, args.change_p)],
                          max_episode_steps=args.max_steps)
    bnn, dyn = make_dirichlet_bnn(grid.n_states, grid.n_actions, grid=grid)
    ck_dir = pathlib.Path(BRIDGE_DATA) / grid.name
    fn = ckpt_name(grid, args.original_p)
    bnn.load(ck_dir, filename=fn)
    init_state = copy.deepcopy(bnn.state_dict())

    log(f"grid={grid.name} {grid.nrow}x{grid.ncol} K={grid.k_dir} "
        f"| n_unfrozen={args.n_unfrozen} alpha={args.cvar_alpha:g} "
        f"| ckpt={fn} (original p={args.original_p:g} "
        f"={[round(x,3) for x in grid.slip_dist(args.original_p)]}) "
        f"-> p'={args.change_p:g} ={[round(x,3) for x in grid.slip_dist(args.change_p)]} @ts=0 "
        f"| score_gamma={GAMMA} plan_gamma={args.plan_gamma:g} "
        f"step_cost={args.step_cost:g} adaptive_alpha={args.adaptive_alpha} "
        f"trial_len={args.max_steps} hole=0-scored "
        f"| K_FORGET={K_FORGET} retrain(lr={RETRAIN_LR},steps={args.retrain_steps}) "
        f"| clip_delta_n={args.clip_delta_n:g}"
        + (" (OFF)" if args.clip_delta_n <= 0 else "")
        + f" | H={args.horizon if args.horizon is not None else 'default(6)'}"
        + f" | conc_prior={args.conc_prior if args.conc_prior else 'symmetric'}"
        + f" | max_forgets={args.max_forgets if args.max_forgets is not None else 'uncapped'}"
        + f" | forget_mode={'ANCHOR (affine pull)' if args.anchor_forget else 'shrink'}"
        + (f" | CONST_RHO={args.const_rho:g} (surprise stat REMOVED)"
           if args.const_rho is not None else "")
        + (f" trigger_rate={args.trigger_rate:g}" if args.trigger_rate is not None else "")
        + (f" | pi_kappa={args.pi_kappa:g}" if args.pi_kappa > 0 else ""))

    trace_path = out_dir / f"{tag}_trace.jsonl"
    trace_fh = open(trace_path, "w") if args.trace else None
    try:
        records = run_trials(args, grid, bnn, dyn, env, init_state, log, trace_fh)
    finally:
        if trace_fh is not None:
            trace_fh.close()

    rets = np.array([r["ret"] for r in records], dtype=float)
    goals = np.array([r["outcome"] == "goal" for r in records], dtype=float)
    fd = [r["first_detect_dbar"] for r in records if r["first_detect_dbar"] is not None]
    sem = float(rets.std(ddof=1) / np.sqrt(len(rets))) if len(rets) > 1 else 0.0
    out = dict(
        grid=grid.name, oracle=args.oracle, n_unfrozen=args.n_unfrozen, cvar_alpha=args.cvar_alpha,
        original_p=args.original_p, change_p=args.change_p,
        original_dist=grid.slip_dist(args.original_p),
        post_change_dist=grid.slip_dist(args.change_p),
        forget=not args.no_forget, counts=not args.no_counts,
        gamma=GAMMA, trials=args.trials, seed=args.seed,
        trial_len=args.max_steps, k_forget=K_FORGET,
        retrain_lr=RETRAIN_LR, retrain_steps=args.retrain_steps, hole_scoring=0,
        clip_delta_n=args.clip_delta_n, horizon=args.horizon,
        planner=args.planner, mcts_sims=args.mcts_sims,
        plan_gamma=args.plan_gamma, step_cost=args.step_cost,
        adaptive_alpha=args.adaptive_alpha,
        conc_prior=args.conc_prior, conc_strength=args.conc_strength,
        max_forgets=args.max_forgets,
        anchor_forget=args.anchor_forget, const_rho=args.const_rho,
        trigger_rate=args.trigger_rate, pi_kappa=args.pi_kappa,
        mean_return=float(rets.mean()), sem_return=sem,
        goal_rate=float(goals.mean()),
        mean_steps=float(np.mean([r["steps"] for r in records])),
        trunc_rate=float(np.mean([r["outcome"] == "truncated" for r in records])),
        hole_rate=float(np.mean([r["outcome"] == "hole" for r in records])),
        first_detect_dbar_mean=float(np.mean(fd)) if fd else float("nan"),
        first_detect_dbar_sem=(float(np.std(fd, ddof=1) / np.sqrt(len(fd)))
                               if len(fd) > 1 else 0.0),
        first_detect_n_trials=len(fd),
        records=records,
    )
    (out_dir / f"{tag}.json").write_text(json.dumps(out, indent=2))
    log(f"saved: return {rets.mean():.3f} +/- {sem:.3f} | goal {goals.mean():.3f} "
        f"| hole {out['hole_rate']:.3f} | trunc {out['trunc_rate']:.3f} "
        f"| dbar@detect {out['first_detect_dbar_mean']:.3f} ({len(fd)}/{args.trials})")


if __name__ == "__main__":
    main()
