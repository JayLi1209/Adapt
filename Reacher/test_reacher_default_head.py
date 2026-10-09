"""CLAUDE.md DEFAULT reacher setting: frozen-body adapter head under non-linear wind.

SETTING
  Reacher-v5, wind intensity 0 -> F_max at ts 0 (CLAUDE.md reacher default), the
  change never announced (the agent gets `notify_change()` only).  The wind is a
  STATIC-per-episode, spatially-varying force field on the fingertip:

      F_x(x,y) = F_max * tanh( sin(2 k_s x + C_x1) + sin(pi k_s y + C_x2) )
      F_y(x,y) = F_max * tanh( sin(2 k_s y + C_y1) + sin(pi k_s x + C_y2) )

  with (C_x1, C_x2, C_y1, C_y2) ~ U[0, 2pi) drawn once per episode and held.
  Because the field is fixed within an episode it is a function of POSITION
  only -- which is what makes it learnable by a state-conditioned adapter, and
  is the whole reason this is a sensible test of the head.

  Checkpoint data/reacher, pretrained on the WIND-FREE env (gear/mass/damping
  x1.0).  Planner CEM+CVaR, H=15, 8 iters, 500 candidates, elite 10%, K=3
  posterior draws, alpha=0.8, gamma=1.0 (no discount), learned reward head,
  manifold projection on.  Per-DIMENSION surprise score throughout.  Episode =
  Reacher's native 50 steps.  k_forget=1.

ARMS  (mirroring Pendulum/test_pendulum_default_head.py)
  no_adapt     frozen pretrained model.  Floor.
  forget_elbo  the existing scheme: per-dim surprise -> PerDimDriftFilter ->
               retention-mode inflation of the head rows -> whole-network ELBO
               refit every step.
  head         the new way.  Body COMPLETELY frozen (mean and variance).
               mu(s,u) = mu_BNN(s,0) + h_phi(s,u), h_phi Bayesian, warm-started
               so h(s,u) = W0 @ u reproduces the body at step 0.  Retention
               inflation is applied to the ADAPTER, not the body.
  head_noforget  ablates the retention step from the head.
  oracle       true windy MuJoCo in the rollout (planning.oracle_cem with the
               same field).  Ceiling.

WHY THERE IS NO head_lin ARM.  The pendulum's LinearActionHead owns the ACTION
channel: mu = mu_BNN(s,0) + w*u.  A mass change lives entirely in that channel,
so the linear head is exact there.  This wind does NOT live in the action
channel at all -- it is an additive, position-dependent force independent of u,
so no multiple of u can express it and a linear action head is structurally
blind to it by construction.  Including it would be a guaranteed-null arm; the
state-conditioned nonlinear head is the only member of the family that spans
this change.  (`--arms` can still request it if you want to see the null.)

Run:  python3 test_reacher_default_head.py --fmax 500 --trials 3
"""
import argparse, json, pathlib, time
import numpy as np, torch, mujoco

from config import device
from bnn import make_gaussian_bnn, load_arch
from bnn.gaussian_workflow import (forget_gaussian_perdim, anchor_head_rows_to_current,
                                   surprise_gaussian)
from bnn.gain_adapter import NonlinearAdapterHead, measure_gain, elbo_step_adapter
from bnn.action_adapter import ActionInterfaceAdapter
from bnn.linear_adapter import LinearAdapterHead, elbo_step_linear_adapter
from drift.filters import PerDimDriftFilter
from env.reacher import (build_reacher_env, wind_force, fingertip_xy,
                         project_reacher_obs, actuator_saturate, action_rotate, OBS_DIM,
                         ACT_DIM, DIST_IDX, EP_LEN, WIND_KS)
from planning.continuous_cem import ContinuousCEMAgent
from planning.oracle_cem import MuJoCoOracleCEM, get_state

MODEL_DIR = "data/reacher"
H, CEM_ITERS, CANDIDATES, K_MODELS, ALPHA, GAMMA = 15, 8, 500, 3, 0.8, 1.0
TRIAL_LEN, K_FORGET, SEED_BASE = EP_LEN, 1, 1000
OUT_SIZE = OBS_DIM + 1                 # 11 model output channels
N_SURP = OBS_DIM                       # surprise scored on state dims only
ACTIVE = list(range(OBS_DIM))
# ── qvel-restricted adaptation ────────────────────────────────────────────────
# Torque enters qpos only at O(dt^2), so of Reacher's 10 state dims exactly TWO
# -- the joint velocities, obs dims 6 and 7 -- carry first-order information
# about an actuator-side fault.  Measured at theta=60: those two dims degrade
# 178x and 7.7x (to 93-96% relative error) while every other dim degrades only
# 2.8-5.6x and stays at 11-21%.
#
# Consequences of NOT restricting:
#   * the uniform ELBO averages 10 per-dim NLLs, so 8 of them ask the head to
#     re-fit channels the frozen model already predicts well;
#   * the calibrated surprise (E[delta]=1 per dim) is averaged over 10 dims, so a
#     genuine 2-dim excess is diluted ~5x.  That is consistent with forget_elbo
#     reporting dbar = 0.39 -- below the rho=1 no-op threshold -- while still
#     firing on ~200 steps through the per-dim deadband gate.
#
# QVEL_DIMS restricts BOTH: the ELBO's active set and the surprise statistic,
# which then becomes chi^2_2 / 2 (two dims, still unit-mean under calibration).
QVEL_DIMS = [6, 7]
# Retrain / forget cadence.  Defaults reproduce the previous behaviour
# (K_RETRAIN=1 -> ELBO every step; MAX_FORGETS=None -> forget every step).
# Rationale for decoupling them: forgetting fired 200x per episode against a
# bounded Adam step, so inflation won every exchange and |W-anchor| stayed
# pinned near its init.  Making retrain LESS frequent and forgetting RARE
# inverts that ratio.
K_RETRAIN = 1              # ELBO update every K_RETRAIN steps
MAX_FORGETS = None         # cap on forget APPLICATIONS per episode (None = no cap)
RETRAIN_LR, RETRAIN_STEPS, N_MC, BETA, Q_MAX = 1e-3, 5, 3, 1.0, 2.0
ADAPT_HID, ADAPT_LR, ADAPT_PRIOR = 64, 3e-3, 0.1
ADAPT_STEPS, ADAPT_MC, ADAPT_BATCH = 5, 3, 256
# lin1_* are the ONE-LAYER head.  Because a backprop+ELBO head is not conjugate,
# "forgetting" has no closed-form meaning any more, so the three lin1 arms are
# the three honest readings of it -- see bnn/linear_adapter.py.
ARMS_DEFAULT = ["no_adapt", "forget_elbo", "retrain", "lin1_noforget",
                "lin1_discount", "head", "oracle"]
NL_ARMS = ("nl1_noforget", "nl2_noforget", "nl1_forget", "nl2_forget",
           "nl1_drift",
           # qvel-restricted: ELBO active set AND surprise both on dims 6,7 only
           "nl1_noforget_qv", "nl1_forget_qv",
           # _nlonly: optimiser sees ONLY the new non-linear layer (l_nl).  The
           # warm-started skip branch is frozen along with the whole BNN body, so
           # the ONLY trainable parameters in the entire model are l_nl's 260.
           "nl1_forget_qv_nlonly",
           # _body: the qvel-restricted head AND a full-network retrain of the
           # frozen BNN.  Everything else identical to nl1_forget_qv.
           "nl1_forget_qv_body",
           # _mul: MULTIPLICATIVE composition, delta = h(s,u) * mu_body(s,0),
           # instead of the additive delta = mu_body(s,0) + h(s,u).  Everything
           # else identical to nl1_forget_qv.
           "nl1_forget_qv_mul",
           # all-dims multiplicative counterparts of nl1_forget / nl1_noforget.
           # mul_dims still restricts the GAIN to qvel (dims 4,5 have a
           # structurally zero delta and admit no gain); what "_qv" controls is
           # the SURPRISE + ELBO active set, which these leave unrestricted.
           "nl1_forget_mul", "nl1_noforget_mul",
           # FORGET-ONLY: inflation fires every step but the head's ELBO never
           # runs, so the head MEAN stays at its warm start h(s,u)=W0 u and only
           # its posterior WIDTH moves.  The mirror image of nl1_noforget (ELBO,
           # no forgetting); together they decompose SFIR into its two halves.
           "nl1_forget_noelbo",
           # NEITHER knob: warm-started head that never forgets AND never fits.
           # The paper table's "no forget + no retrain" cell, and the analogue of
           # Pendulum's no_forget_no_retrain -- it should reproduce no_adapt to
           # within planner noise, which makes it the grid's validity check.
           "nl1_noforget_noelbo")
# ACTION-INTERFACE arm: no output-side head at all.  Plans with the UNTOUCHED
# frozen model and emits R_hat^T a*, with R_hat identified in closed form.
ACT_ARMS = ("act_rot",)
LIN1_ARMS = ("lin1_noforget", "lin1_drift", "lin1_discount", "lin1_precision",
             # ablation arms: three ways to stop inflation from erasing the ELBO
             "lin1_rhofloor",   # (1) bound rho: cap per-call erasure at 10%
             "lin1_l2",         # (2) forgetting as an L2-to-anchor LOSS term
             "lin1_softdecay",  # (3) soft additive decay, fixed 2% per call
             "lin1_steps50",    # (4) 50 Adam steps/timestep instead of 5
             "lin1_l2_steps50", # (2)+(4) combined
             )
RHO_FLOOR = 0.9            # (1) at most 10% of the correction erased per call
SOFT_DECAY = 0.02          # (3) fixed 2% pull toward the anchor per call
L2_ANCHOR = 1e-3           # (2) weight on ||mu - anchor||^2
STEPS_MANY = 50            # (4)
# act_rot: identify after this many transitions, then refit every ACT_REFIT_EVERY
# steps.  Four unknowns need very little data -- the estimate is usable by ~30
# samples -- and re-solving is a 4x4 solve, so the cadence is cheap.
ACT_MIN_SAMPLES = 25
ACT_REFIT_EVERY = 25
ACT_REFIT_ITERS = 3
DISCOUNT_FLOOR = 1e-3


def load_model():
    hid, nl = load_arch(MODEL_DIR)
    bnn, dyn = make_gaussian_bnn(OBS_DIM, ACT_DIM, hid_size=hid, num_layers=nl)
    dyn.input_normalizer.load(MODEL_DIR)
    bnn.load(MODEL_DIR, "bnn_dynamics.pth")
    bnn.eval()
    # MuJoCo is deterministic: the aleatoric channel is absorbed model error, not
    # process noise, so it is not re-injected at every imagined step.
    bnn.aleatoric_in_rollout = False
    return bnn, dyn


def gain_probe(n=3000, seed=3):
    """States for measure_gain: the reachable workspace, matching pretraining."""
    from pretrain_reacher import collect_data, to_arrays
    o, _a, _n, _r = to_arrays(collect_data(n, np.random.default_rng(seed)))
    return torch.tensor(o, device=device)


def oracle_reward_fn(nq):
    def f(states, acts):
        q = states[:, :, 1:1 + nq]
        fx, fy = fingertip_xy(q[:, :, 0], q[:, :, 1])
        d = np.sqrt((fx - q[:, :, 2]) ** 2 + (fy - q[:, :, 3]) ** 2)
        return -d - (acts ** 2).sum(axis=2)
    return f


def run_trial(arm, seed, bnn, dyn, init_state, W0, fmax, wind_ks,
              trial_len=TRIAL_LEN, alpha=ALPHA, k_models=K_MODELS, sat=None,
              rot=None, k_retrain=K_RETRAIN, max_forgets=MAX_FORGETS,
              save_traces=False):
    bnn.load_state_dict(init_state)
    # SCOPE OF THE ONLINE ELBO REFIT.  These two arms differ ONLY in which
    # variational parameters the optimiser is allowed to move; the surprise ->
    # smooth -> inflate stage is identical.
    #   forget_elbo : the OUTPUT (head) layer only -- bayes_layers[-1].  This is
    #                 the same set of rows the retention inflation acts on, so
    #                 forget and refit operate on exactly the same parameters.
    #   retrain     : the ENTIRE network, every Bayesian layer (~281k params).
    if arm == "forget_elbo":
        _tune = [bnn.bayes_layers[-1]]
    elif arm in ("retrain", "nl1_forget_qv_body"):
        _tune = list(bnn.bayes_layers)
    else:
        _tune = []
    opt = (torch.optim.Adam([p for l in _tune
                             for p in (l.weight_mu, l.bias_mu, l.weight_rho, l.bias_rho)],
                            lr=RETRAIN_LR) if _tune else None)
    # max_episode_steps: CLAUDE.md says experiments run with NO TRUNCATION, and
    # here it is load-bearing -- an online adapter that only ever sees Reacher's
    # native 50 steps gets 50 samples to learn a 2-D force field from.
    sat_g_sched = sat_k_sched = None
    if sat is not None:
        g0, g1, a0, a1 = sat
        sat_g_sched = [(0, g0), (trial_len, g1)]
        sat_k_sched = [(0, a0), (trial_len, a1)]
    rot_sched = [(0, rot), (trial_len, rot)] if rot is not None else None
    env = build_reacher_env(wind_schedule=[(0, fmax)] if fmax > 0 else None,
                            wind_ks=wind_ks,
                            sat_gain_schedule=sat_g_sched,
                            sat_knee_schedule=sat_k_sched,
                            rot_schedule=rot_sched,
                            max_episode_steps=trial_len + 1)
    torch.manual_seed(seed + 10000)
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped

    if arm == "oracle":
        fid = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_BODY, "fingertip")
        offs, ks = env.wind_offsets, env.wind_ks
        wf = (lambda qp: wind_force(*fingertip_xy(qp[:, 0], qp[:, 1]), fmax, ks, offs)
              ) if fmax > 0 else None
        cf = None
        if rot is not None and sat is None:
            def cf(a, h, _r=rot):
                th = np.deg2rad(_r); c, s_ = np.cos(th), np.sin(th)
                R = np.array([[c, -s_], [s_, c]])
                return a @ R.T          # (B,2) candidates -> rotated torques
        elif sat is not None:
            def cf(a, h, _e=env):
                # The oracle knows the true actuator map, including how it will
                # have drifted h steps into the imagined future.
                g, al = _e._sat_params(_e._step_count + h)
                return actuator_saturate(a, g, al)
        agent = MuJoCoOracleCEM(u.model, u.frame_skip, oracle_reward_fn(u.model.nq),
                                ACT_DIM, horizon=H, n_cem_iters=CEM_ITERS,
                                n_candidates=CANDIDATES, gamma=GAMMA,
                                rng=np.random.default_rng(seed), nthread=12,
                                wind_fn=wf, wind_bodies=[fid] if fmax > 0 else None,
                                ctrl_fn=cf)
    else:
        agent = ContinuousCEMAgent(dyn, bnn, OBS_DIM, ACT_DIM, horizon=H,
                                   n_cem_iters=CEM_ITERS, n_candidates=CANDIDATES,
                                   k_models=k_models, cvar_alpha=alpha, gamma=GAMMA,
                                   action_low=-1.0, action_high=1.0,
                                   obs_project=project_reacher_obs,
                                   rng=np.random.default_rng(seed))
    agent.reset(); agent.notify_change()
    pdf = PerDimDriftFilter(n_dims=N_SURP); pdf.reset()

    nnhead = aopt = None
    if arm in LIN1_ARMS:
        nnhead = LinearAdapterHead(W0, obs_dim=OBS_DIM, act_dim=ACT_DIM,
                                   prior_std=ADAPT_PRIOR).to(device)
        nnhead.attach(dyn)
        aopt = torch.optim.Adam(nnhead.parameters(), lr=ADAPT_LR)
    elif arm in ("head", "head_noforget", "head_meanroll") or arm in NL_ARMS:
        # nl1 / nl2: ONE- vs TWO-layer NON-LINEAR branch, both with the skip
        # branch and the same warm start.  nl*_noforget do NOT forget, so the
        # comparison isolates DEPTH alone.
        _depth = 1 if arm in ("nl1_noforget", "nl1_forget", "nl1_drift",
                              "nl1_noforget_qv", "nl1_forget_qv",
           # _nlonly: optimiser sees ONLY the new non-linear layer (l_nl).  The
           # warm-started skip branch is frozen along with the whole BNN body, so
           # the ONLY trainable parameters in the entire model are l_nl's 260.
           "nl1_forget_qv_nlonly",
           # _body: the qvel-restricted head AND a full-network retrain of the
           # frozen BNN.  Everything else identical to nl1_forget_qv.
           "nl1_forget_qv_body", "nl1_forget_qv_mul",
           "nl1_forget_mul", "nl1_noforget_mul",
           "nl1_forget_noelbo", "nl1_noforget_noelbo") else 2
        _compose = "mul" if arm.endswith("_mul") else "add"
        nnhead = NonlinearAdapterHead(W0, obs_dim=OBS_DIM, hid=ADAPT_HID,
                                      prior_std=ADAPT_PRIOR, act_dim=ACT_DIM,
                                      depth=_depth, compose=_compose).to(device)
        if _compose == "mul":
            # Gain acts on the qvel block ONLY; all other dims pass through as
            # the body's own mu_body(s,0).  Restricting it here keeps the mul
            # arm on exactly the subspace the fault lives in (measured: qvel
            # carries 98.7% of the wind's added one-step error) and avoids dims
            # 4,5 whose delta is structurally zero and where a gain has no
            # leverage.
            nnhead.mul_dims = list(QVEL_DIMS)
        # ROLLOUT POSTERIOR MODE.  The planner draws K=3 posterior samples per
        # candidate, and the adapter is sampled along with the body.  Under
        # retention inflation that is destructive here: measured on this env,
        # forgetting drives the skip/l2 sigmas from 0.005 to 0.08 (16x) and the
        # head's spread across draws reaches 3.49 while the correction it is
        # making is only |h| ~ 1.36 -- the imagined trajectories are then mostly
        # adapter noise, and CVaR is averaging a tail that carries no
        # information about the plant.  `head_meanroll` keeps the SAME learned
        # posterior (so forgetting still governs learning) but rolls out with the
        # adapter's MEAN, leaving the body to supply the epistemic spread CVaR
        # consumes.
        nnhead.rollout_mean_only = (arm == "head_meanroll")
        nnhead.attach(dyn)
        if arm.endswith("_nlonly"):
            # Freeze the warm-started skip branch: only l_nl is optimised, so the
            # head keeps h(s,u) = W0 @ u exactly as its linear part and the new
            # non-linear layer alone owns the correction.
            for _pn, _pp in nnhead.named_parameters():
                if _pn.startswith("skip."):
                    _pp.requires_grad_(False)
            aopt = torch.optim.Adam([_pp for _pp in nnhead.parameters()
                                     if _pp.requires_grad], lr=ADAPT_LR)
        else:
            aopt = torch.optim.Adam(nnhead.parameters(), lr=ADAPT_LR)

    actad = None
    if arm in ACT_ARMS:
        # Four unknowns on the qvel rows.  No head is attached: `dyn` and `bnn`
        # stay exactly as pretrained, so the planner, the reward channel and
        # every surprise statistic keep their calibrated meaning.
        actad = ActionInterfaceAdapter(ACT_DIM, OBS_DIM, QVEL_DIMS,
                                       min_samples=ACT_MIN_SAMPLES)

    buf_in, buf_act, buf_tgt = [], [], []
    rew, dists, perr, dbar_tr, n_forget = [], [], [], [], 0
    for step in range(trial_len):
        a = agent.act(get_state(env)) if arm == "oracle" else agent.act(obs)
        aa = np.asarray(a, np.float32).ravel()
        # `aa` is the PLANNED action -- what the frozen model was queried with,
        # and therefore what identification must regress on.  `a_env` is what the
        # plant receives.  For act_rot they differ by R_hat^T; because that is
        # orthogonal, ||a_env|| == ||aa||, so the control cost the reward charges
        # is unchanged and the reward identity survives.
        a_env = actad.emit(aa) if actad is not None else a
        o_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        a_t = torch.as_tensor(aa, dtype=torch.float32, device=device)
        with torch.no_grad():
            zero = torch.zeros_like(a_t.unsqueeze(0))
            m0, lv0 = bnn._run_network(dyn._get_model_input(o_t.unsqueeze(0), zero),
                                       sample=False)
            if nnhead is not None:
                pm, _ = nnhead.predict(bnn, dyn, o_t.unsqueeze(0), a_t.unsqueeze(0))
            else:
                pm, _ = bnn._run_network(
                    dyn._get_model_input(o_t.unsqueeze(0), a_t.unsqueeze(0)), sample=False)
        nxt, r, _te, _tr, _info = env.step(a_env)
        rew.append(float(r)); dists.append(float(np.linalg.norm(nxt[DIST_IDX])))
        n_t = torch.as_tensor(nxt, dtype=torch.float32, device=device)
        perr.append(float(np.linalg.norm(
            (o_t + pm[0, :OBS_DIM]).cpu().numpy() - np.asarray(nxt, np.float64))))

        if arm in ("forget_elbo", "retrain", "head", "head_noforget",
                   "head_meanroll") or arm in LIN1_ARMS or arm in NL_ARMS:
            if nnhead is not None:
                with torch.no_grad():
                    ms = torch.stack([nnhead.predict(bnn, dyn, o_t.unsqueeze(0),
                                                     a_t.unsqueeze(0), sample=True
                                                     )[0][:, :OBS_DIM] for _ in range(8)])
                    lvs = nnhead.predict(bnn, dyn, o_t.unsqueeze(0), a_t.unsqueeze(0),
                                         sample=False)[1][:, :OBS_DIM]
                    S = ms.var(0, unbiased=False) + torch.exp(lvs) + 1e-12
                    nu = (n_t - o_t)[:OBS_DIM].view(1, -1) - ms.mean(0)
                dn = (nu.pow(2) / S).squeeze(0).cpu().numpy().astype(np.float64)
            else:
                dn = surprise_gaussian(dyn, bnn, obs, aa, nxt, r,
                                       use_reward_dim=False)["delta_n_vec"]
            pdf.update(dn)
            dbar = np.asarray(pdf.delta_bar, dtype=np.float64)[:N_SURP]
            if ("_qv" in arm):
                # chi^2_2 / 2 : score drift on the qvel block ONLY.  Other dims
                # keep dbar = 1 (the calibrated no-op value) so they neither
                # dilute the statistic nor trigger their own inflation.
                _q = np.ones_like(dbar); _q[QVEL_DIMS] = dbar[QVEL_DIMS]
                dbar = _q
            dbar_tr.append(float(np.mean(dbar[QVEL_DIMS] if ("_qv" in arm)
                                          else dbar)))

        if actad is not None:
            # Record the EMITTED action.  Identification must regress on what the
            # plant actually received: once compensation is live, planned and
            # emitted differ by R_hat^T and that map moves as the estimate
            # updates, so keying the buffer on planned actions mixes several
            # effective maps and the fit drifts away from the truth.
            _ae = torch.as_tensor(np.asarray(a_env, np.float32).ravel(),
                                  dtype=torch.float32, device=device)
            buf_in.append(o_t.unsqueeze(0)); buf_act.append(_ae.unsqueeze(0))
            buf_tgt.append((n_t - o_t).unsqueeze(0))
            if (step + 1) >= ACT_MIN_SAMPLES and (step + 1) % ACT_REFIT_EVERY == 0:
                actad.refit(bnn, dyn,
                            list(zip(buf_in, buf_act,
                                     [b.squeeze(0) for b in buf_tgt])),
                            n_iter=ACT_REFIT_ITERS)

        if nnhead is not None:
            buf_in.append(o_t.unsqueeze(0)); buf_act.append(a_t.unsqueeze(0))
            buf_tgt.append((n_t - o_t).unsqueeze(0))
            _forget_ok = (max_forgets is None or n_forget < max_forgets)
            if (step + 1) % K_FORGET == 0 and _forget_ok:
                if arm == "nl1_drift":
                    # coherent non-conjugate forgetting: decay BOTH moments
                    _rho, fired = nnhead.forget(dbar, mode="prior_drift")
                    if fired:
                        n_forget += 1; nnhead.anchor(include_sigma=True)
                elif arm in ("nl1_forget", "nl2_forget", "nl1_forget_qv",
                             "nl1_forget_qv_nlonly",
           # _body: the qvel-restricted head AND a full-network retrain of the
           # frozen BNN.  Everything else identical to nl1_forget_qv.
           "nl1_forget_qv_body", "nl1_forget_qv_mul",
                             "nl1_forget_mul", "nl1_forget_noelbo"):
                    # surprise -> filter -> inflate, applied to the ADAPTER.
                    # NonlinearAdapterHead.forget uses prior-drift semantics on
                    # its own rows (skip/l2 or skip/l_nl per-dim; l1 shared takes
                    # min rho), so a depth-1 head is fully compatible with the
                    # mechanism -- nothing about SFIR requires two layers.
                    _rho, fired = nnhead.forget(dbar)
                    if fired:
                        n_forget += 1; nnhead.anchor(include_sigma=True)
                elif arm in ("head", "head_meanroll"):
                    _rho, fired = nnhead.forget(dbar)
                    if fired:
                        n_forget += 1; nnhead.anchor(include_sigma=True)
                elif arm == "lin1_rhofloor":
                    _rho, fired = nnhead.forget(dbar, mode="prior_drift",
                                                rho_floor=RHO_FLOOR)
                    if fired:
                        n_forget += 1; nnhead.anchor(include_sigma=True)
                elif arm == "lin1_softdecay":
                    _rho, fired = nnhead.forget(dbar, mode="prior_drift",
                                                decay_step=SOFT_DECAY)
                    if fired:
                        n_forget += 1; nnhead.anchor(include_sigma=True)
                elif arm in ("lin1_l2", "lin1_l2_steps50", "lin1_steps50"):
                    pass          # no parameter reset; see the ELBO call below
                elif arm in ("lin1_drift", "lin1_discount"):
                    # prior-drift: decay BOTH moments toward the warm start.
                    # lin1_discount now INFLATES too, so it differs from
                    # lin1_noforget in exactly ONE thing: whether forgetting is
                    # applied.  Both share the identical uniform-weight ELBO
                    # below, since uniform w beat rho^age weighting
                    # (paired -2.79 vs -5.09 under saturation).
                    _rho, fired = nnhead.forget(dbar, mode="prior_drift")
                    if fired:
                        n_forget += 1; nnhead.anchor(include_sigma=True)
                elif arm == "lin1_precision":
                    # legacy conjugate formula on a non-conjugate head: inflate
                    # precision, leave the mean.  Included to MEASURE the
                    # incoherence rather than assert it.
                    _rho, fired = nnhead.forget(dbar, mode="precision_only")
                    if fired:
                        n_forget += 1; nnhead.anchor(include_sigma=True)
            n_buf = len(buf_in)
            idx = (torch.randint(0, n_buf, (ADAPT_BATCH,), device=device)
                   if n_buf > ADAPT_BATCH else torch.arange(n_buf, device=device))
            # nl1_forget_noelbo: forgetting only -- never fit the head.
            _do_retrain = ((step + 1) % k_retrain == 0
                           and arm not in ("nl1_forget_noelbo",
                                           "nl1_noforget_noelbo"))
            if not _do_retrain:
                pass
            elif arm in LIN1_ARMS:
                # SAME update for every lin1 arm: uniform-weight ELBO.  The
                # rho^age sample weighting that lin1_discount used previously is
                # gone -- it lost to uniform weighting on both perturbations, so
                # the arms now differ only in their FORGETTING, not their fit.
                _steps = (STEPS_MANY if arm in ("lin1_steps50", "lin1_l2_steps50")
                          else ADAPT_STEPS)
                # L2 arms: forgetting lives in the OBJECTIVE.  Scale the pull per
                # dim by that dim's rho, so a surprising dim is held loosely and a
                # quiet one is held tight -- the same information the
                # multiplicative rule used, applied as a gradient instead.
                _l2 = L2_ANCHOR if arm in ("lin1_l2", "lin1_l2_steps50") else 0.0
                _sc = None
                if _l2 > 0.0:
                    _sc = np.clip(1.0 / np.maximum(dbar, 1.0), 1e-3, 1.0)
                _act = QVEL_DIMS if ("_qv" in arm) else ACTIVE
                elbo_step_linear_adapter(
                    nnhead, aopt, bnn, dyn, torch.cat(buf_in)[idx],
                    torch.cat(buf_act)[idx], torch.cat(buf_tgt)[idx], _act,
                    n_steps=_steps, n_mc=ADAPT_MC, beta=BETA,
                    kl_denom=n_buf, sample_w=None, l2_anchor=_l2, l2_scale=_sc)
            elif True:
                _st = STEPS_MANY if arm in NL_ARMS else ADAPT_STEPS
                _act = QVEL_DIMS if ("_qv" in arm) else ACTIVE
                elbo_step_adapter(nnhead, aopt, bnn, dyn, torch.cat(buf_in)[idx],
                                  torch.cat(buf_act)[idx], torch.cat(buf_tgt)[idx],
                                  _act, n_steps=_st, n_mc=ADAPT_MC,
                                  beta=BETA, kl_denom=n_buf)

        if arm == "nl1_forget_qv_body" and (step + 1) % k_retrain == 0:
            # FULL-NETWORK retrain running alongside the head.  The body is fit
            # on the COMPOSED residual it is actually responsible for: the head
            # already supplies h(s,u), so the body's target is the delta minus
            # the head's current contribution.  Without that subtraction the two
            # would both chase the whole delta and double-count it.
            _n = len(buf_in)
            _ix = (torch.randint(0, _n, (ADAPT_BATCH,), device=device)
                   if _n > ADAPT_BATCH else torch.arange(_n, device=device))
            _ob, _ab = torch.cat(buf_in)[_ix], torch.cat(buf_act)[_ix]
            _tb = torch.cat(buf_tgt)[_ix]
            with torch.no_grad():
                _h = nnhead.forward(torch.cat([_ob, _ab], dim=-1), sample=False)
            _tgt = torch.cat([_tb[:, :OBS_DIM] - _h,
                              torch.zeros(_tb.shape[0], 1, device=device)], 1)
            for _ in range(RETRAIN_STEPS):
                opt.zero_grad()
                _loss, _m = bnn.loss(dyn._get_model_input(_ob, torch.zeros_like(_ab)),
                                     _tgt)
                if torch.isfinite(_loss):
                    _loss.backward()
                    torch.nn.utils.clip_grad_norm_(bnn.parameters(), 10.0)
                    opt.step()

        if arm in ("forget_elbo", "retrain"):
            if (step + 1) % K_FORGET == 0:
                _q, fired = forget_gaussian_perdim(bnn, pdf, out_size=OUT_SIZE,
                                                   inflate_mode="retention", q_max=Q_MAX)
                if np.asarray(fired).any():
                    n_forget += 1
                    anchor_head_rows_to_current(bnn, np.where(fired)[0], OUT_SIZE, True)
            buf_in.append(o_t.unsqueeze(0)); buf_act.append(a_t.unsqueeze(0))
            buf_tgt.append((n_t - o_t).unsqueeze(0))
            n_buf = len(buf_in)
            idx = (torch.randint(0, n_buf, (ADAPT_BATCH,), device=device)
                   if n_buf > ADAPT_BATCH else torch.arange(n_buf, device=device))
            ob, ab = torch.cat(buf_in)[idx], torch.cat(buf_act)[idx]
            tb = torch.cat(buf_tgt)[idx]
            tgt_full = torch.cat([tb, torch.zeros(tb.shape[0], 1, device=device)], 1)
            for _ in range(RETRAIN_STEPS):
                opt.zero_grad()
                loss, _meta = bnn.loss(dyn._get_model_input(ob, ab), tgt_full)
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(bnn.parameters(), 10.0)
                    opt.step()
        obs = nxt

    if nnhead is not None:
        nnhead.detach()
    out = dict(arm=arm, seed=seed, ret=float(np.sum(rew)),
                final_dist=float(dists[-1]), last10=float(np.mean(dists[-10:])),
                pred_err=float(np.mean(perr)), n_forget=int(n_forget),
                dbar=float(np.mean(dbar_tr)) if dbar_tr else float("nan"),
                est_deg=(float(actad.angle_deg) if actad is not None
                         else float("nan")))
    # PER-STEP TRACES (--save-traces).  rew/dists/perr are already accumulated
    # every step for the scalar summaries above; this just keeps them so the
    # cumulative-return curve can be plotted instead of only its endpoint.
    # Off by default: 200 floats x 3 x arms x seeds bloats the JSON.
    if save_traces:
        out["rew_t"] = [float(x) for x in rew]
        out["dist_t"] = [float(x) for x in dists]
        out["perr_t"] = [float(x) for x in perr]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmax", type=float, default=500.0)
    ap.add_argument("--k-retrain", type=int, default=K_RETRAIN,
                    help="run the ELBO update every K steps (1 = every step)")
    ap.add_argument("--max-forgets", type=int, default=None,
                    help="cap the number of forget APPLICATIONS per episode")
    ap.add_argument("--rot", type=float, default=None,
                    help="action-space rotation theta in DEGREES: tau = R(theta) a, "
                         "a transmission-coupling fault.  R is orthogonal so the "
                         "torque budget is unchanged and the whole gap is direction "
                         "error.  Fixed from ts 0.")
    ap.add_argument("--sat", type=str, default=None,
                    help="actuator saturation tau = g*tanh(a/alpha).  Give TWO "
                         "values 'g,alpha' for a FIXED map held constant for the "
                         "whole episode (a step change at ts 0, like the wind), or "
                         "FOUR 'g0,g1,alpha0,alpha1' to drift them linearly.")
    ap.add_argument("--wind-ks", type=float, default=WIND_KS)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--trial-len", type=int, default=TRIAL_LEN,
                    help="steps per episode; Reacher's native limit is 50, which "
                         "is a very short adaptation window (CLAUDE.md: no truncation)")
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--arms", type=str, default=",".join(ARMS_DEFAULT))
    ap.add_argument("--alpha", type=float, default=ALPHA,
                    help="CVaR tail fraction over the K posterior draws; "
                         "1.0 = risk-neutral mean, smaller = more risk-averse")
    ap.add_argument("--k-models", type=int, default=K_MODELS,
                    help="posterior draws per candidate; CVaR is a no-op at K=1")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--save-traces", action="store_true",
                    help="record per-timestep reward/distance/pred_err in the "
                         "JSON so cumulative-return curves can be plotted")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]

    _sat = None
    if args.sat:
        _v = tuple(float(x) for x in args.sat.split(","))
        if len(_v) == 2:            # FIXED: g and alpha constant all episode
            _sat = (_v[0], _v[0], _v[1], _v[1])
        elif len(_v) == 4:          # drifting endpoints
            _sat = _v
        else:
            raise SystemExit("--sat needs 'g,alpha' (fixed) or "
                             "'g0,g1,alpha0,alpha1' (drifting)")
    bnn, dyn = load_model()
    init_state = {k: v.detach().clone() for k, v in bnn.state_dict().items()}
    W0 = measure_gain(bnn, dyn, gain_probe(), n_dims=OBS_DIM, u=1.0, act_dim=ACT_DIM)

    print(f"\n=== Reacher non-linear wind: frozen body + adapter head ===")
    print(f"  wind      : F_max 0 -> {args.fmax} at ts 0, k_s={args.wind_ks}, "
          f"static per-episode phases")
    if args.rot is not None:
        print(f"  rotation  : tau = R({args.rot} deg) @ a  (transmission-coupling "
              f"fault; ||tau||==||a||, cos={np.cos(np.deg2rad(args.rot)):.3f})")
    if _sat:
        _fixed = (_sat[0] == _sat[1] and _sat[2] == _sat[3])
        if _fixed:
            print(f"  saturation: tau = {_sat[0]}*tanh(a/{_sat[2]}), FIXED at ts 0 "
                  f"and held constant for all {args.trial_len} steps")
        else:
            print(f"  saturation: tau=g_t*tanh(a/alpha_t), g {_sat[0]}->{_sat[1]}, "
                  f"alpha {_sat[2]}->{_sat[3]} drifting over {args.trial_len} steps")
    print(f"  planner   : CEM+CVaR H={H} J={CANDIDATES} iters={CEM_ITERS} "
          f"K={args.k_models} alpha={args.alpha} gamma={GAMMA}")
    print(f"  cadence   : ELBO every {args.k_retrain} step(s) | "
          f"max forgets/episode = {args.max_forgets if args.max_forgets is not None else 'uncapped'}")
    print(f"  episode   : {args.trial_len} steps | k_forget={K_FORGET} | "
          f"{args.trials} seeds ({args.seed_base}..{args.seed_base + args.trials - 1})")
    _nh = NonlinearAdapterHead(W0, obs_dim=OBS_DIM, hid=ADAPT_HID,
                               prior_std=ADAPT_PRIOR, act_dim=ACT_DIM)
    _l1 = LinearAdapterHead(W0, obs_dim=OBS_DIM, act_dim=ACT_DIM,
                            prior_std=ADAPT_PRIOR)
    print(f"  heads     : 3-layer {sum(p.numel() for p in _nh.parameters()):,} params"
          f" | 1-layer {sum(p.numel() for p in _l1.parameters()):,} params")
    print(f"  W0 gain   : diag {np.round(np.diag(W0[6:8, :]), 4).tolist()} "
          f"(torque j -> qvel j)")

    rows = []
    for arm in arms:
        t0 = time.time(); per = []
        for tr in range(args.trials):
            out = run_trial(arm, args.seed_base + tr, bnn, dyn, init_state, W0,
                            args.fmax, args.wind_ks, args.trial_len,
                            args.alpha, args.k_models, _sat, args.rot,
                            args.k_retrain, args.max_forgets,
                            args.save_traces)
            per.append(out)
            print(f"    {arm:14s} seed {out['seed']}: return {out['ret']:8.3f}  "
                  f"final_d {out['final_dist']:.4f}  pred_err {out['pred_err']:.4f}"
                  f"  forgets {out['n_forget']}"
                  + (f"  est_rot {out['est_deg']:+.2f}deg"
                     if out['est_deg'] == out['est_deg'] else ""))
        R = np.array([p["ret"] for p in per])
        rows.append(dict(arm=arm, ret=float(R.mean()),
                         ret_se=float(R.std() / max(np.sqrt(len(R)), 1)),
                         final_dist=float(np.mean([p["final_dist"] for p in per])),
                         last10=float(np.mean([p["last10"] for p in per])),
                         pred_err=float(np.mean([p["pred_err"] for p in per])),
                         dbar=float(np.nanmean([p["dbar"] for p in per])),
                         n_forget=float(np.mean([p["n_forget"] for p in per])),
                         trials=per))
        print(f"  {arm:14s} -> return {rows[-1]['ret']:8.3f} +/- {rows[-1]['ret_se']:.3f}"
              f"   [{time.time() - t0:.0f}s]")

    print(f"\n=== SUMMARY (F_max={args.fmax}, {args.trials} seeds) ===")
    print(f"  {'arm':<15}{'return':>12}{'±SE':>8}{'final_d':>10}{'last10_d':>10}"
          f"{'pred_err':>10}{'dbar':>9}")
    for r in rows:
        print(f"  {r['arm']:<15}{r['ret']:>12.3f}{r['ret_se']:>8.3f}"
              f"{r['final_dist']:>10.4f}{r['last10']:>10.4f}{r['pred_err']:>10.4f}"
              f"{r['dbar']:>9.2f}")
    by = {r["arm"]: r for r in rows}
    if "oracle" in by and "no_adapt" in by:
        lo, hi = by["no_adapt"]["ret"], by["oracle"]["ret"]
        print(f"\n  headroom (oracle - no_adapt) = {hi - lo:.3f}")
        for a in ("head", "head_noforget", "forget_elbo"):
            if a in by:
                rec = (by[a]["ret"] - lo) / max(hi - lo, 1e-9)
                print(f"    {a:<15} recovers {100 * rec:6.1f}% of it "
                      f"({by[a]['ret']:.3f})")
    out = pathlib.Path(args.out or f"results/wind_f{int(args.fmax)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(config=vars(args), rows=rows), indent=2))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
