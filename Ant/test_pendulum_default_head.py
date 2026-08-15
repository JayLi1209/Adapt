"""CLAUDE.md DEFAULT pendulum setting, 100 trials: conjugate action head vs the
surprise-forget-inflate scheme it is meant to replace.

SETTING (matches run_pendulum_default.py exactly unless noted)
  mass 1.0 -> 4.0 @ ts 0, gravity 10.0, episode 200, gamma 0.99, Gaussian head,
  |u| <= 2 (Pendulum-v1 default -- build_pendulum_env with no max_torque),
  checkpoint data/pendulum (pretrained on the REAL env at mass 1, |u| <= 2, with
  CLAUDE.md's goal-weighted sampler).  Planner CEM+CVaR, H=40, 8 iters, 500
  candidates, elite 10%, K=10 posterior draws, alpha=1.0, analytic reward
  (planner_reward=analytic, as run_pendulum_default prints).  Per-DIMENSION
  surprise score throughout.  k_forget=5.  100 trials, seeds 1000..1099.

ARMS
  no_adapt     frozen pretrained model.  Floor.
  forget_elbo  the existing scheme at its best known configuration: per-dim
               surprise -> PerDimDriftFilter -> RETENTION-mode inflation of the
               head rows -> whole-network ELBO refit every step.
  head         the new way.  Network COMPLETELY frozen (mean and variance).
               mu(s,u) = mu_BNN(s,0) + w*u, with w from a conjugate Gaussian
               recursion whose discount is the SAME retention function the old
               scheme applied to the weights:
                   Lambda <- lambda*Lambda + u^2/sigma_n^2
                   b      <- lambda*b      + u*r /sigma_n^2
                   lambda  = clip(1/max(delta_bar,1), 1e-3, 1)
               i.e. surprise -> smooth -> discount applied to conjugate
               sufficient statistics instead of to variational weight params, so
               one knob moves the mean (w=b/Lambda) and the variance (1/Lambda)
               together.  Target w[thdot] = 3*dt/(m l^2) = 0.0375.
  oracle       true mass-4 dynamics in the rollout, same warm-starting agent.
               Ceiling.  (The stored oracle numbers in results/ came from a
               cold-solving CEM and are not comparable to warm-started arms.)

Sharded by trial so it can be run as N parallel processes; merge with --merge.
"""
import argparse, copy, json, math, pathlib, time
import numpy as np, torch

from config import device
from env import build_pendulum_env
from bnn.gaussian_model import make_gaussian_bnn, load_arch
from bnn.gaussian_workflow import (forget_gaussian_perdim, retrain_gaussian_full_elbo,
                                   anchor_head_rows_to_current, surprise_gaussian)
from bnn.gain_adapter import (LinearActionHead, NonlinearAdapterHead, measure_gain,
                              surprise_composed, elbo_step_adapter)
from drift.filters import PerDimDriftFilter
from planning.continuous_cem import (ContinuousCEMAgent, project_unit_circle,
                                     pendulum_reward)
import defaults as rpd

MODEL_DIR = "data/pendulum"
MAX_T = 2.0                     # Pendulum-v1 default actuator limit
H, CEM_ITERS, CANDIDATES, ELITE, K_MODELS, ALPHA = 40, 8, 500, 0.1, 10, 1.0
GAMMA, MASS, GRAV = rpd.GAMMA, rpd.TARGET_MASS, rpd.DEFAULT_GRAV
TRIAL_LEN, K_FORGET, SEED_BASE = rpd.TRIAL_LEN, rpd.K_FORGET, 1000
OUT_SIZE, N_SURP, ACTIVE = 4, 3, [0, 1, 2]
RETRAIN_LR, RETRAIN_STEPS, N_MC, BETA, Q_MAX = 1e-3, 5, 3, 1.0, 2.0
UP_ANGLE, UP_SPEED, UP_HOLD = 0.2, 1.0, 20
MAX_SPEED, DT = 8.0, 0.05
W_TARGET = 3.0 * DT / (MASS * 1.0 ** 2)
# "head" is now the NONLINEAR ELBO adapter (it replaced the closed-form linear
# one as the method).  "head_lin" keeps the conjugate linear head as the explicit
# comparison, and "head_noforget" ablates the retention step from the new head.
ARMS = ["no_adapt", "forget_elbo", "head_lin", "head", "head_noforget", "oracle"]
ADAPT_HID, ADAPT_LR, ADAPT_PRIOR = 64, 3e-3, 0.1
ADAPT_STEPS, ADAPT_MC, ADAPT_BATCH = 5, 3, 256


def analytic(th, wd, u, m=None, g=None):
    """Gymnasium Pendulum-v1 dynamics.  Gravity enters ONLY the passive term
    3g/(2l)*sin(theta); the control term 3u/(m l^2) has no g in it.  So a
    gravity change is invisible to any adapter that only owns the action
    channel, and vice versa for a mass change."""
    m = MASS if m is None else m
    g = GRAV if g is None else g
    u = u.clamp(-MAX_T, MAX_T)
    nd = (wd + (3.0 * g / 2.0 * torch.sin(th) + 3.0 / m * u) * DT
          ).clamp(-MAX_SPEED, MAX_SPEED)
    nth = th + nd * DT
    return ((nth + math.pi) % (2 * math.pi)) - math.pi, nd


def gain_probe(n=3000, seed=3):
    g = torch.Generator().manual_seed(seed)
    th = (torch.rand(n, generator=g) * 2 - 1) * math.pi
    wd = (torch.rand(n, generator=g) * 2 - 1) * MAX_SPEED
    return torch.stack([torch.cos(th), torch.sin(th), wd], -1).to(device)


def run_trial(arm, seed, bnn, dyn, init_state, agent, w0):
    bnn.load_state_dict(init_state)
    # Fresh optimiser per trial: Adam's state dict is keyed by Parameter object
    # and carries running moments, so it must not leak across independent trials.
    opt = torch.optim.Adam([p for l in bnn.bayes_layers
                            for p in (l.weight_mu, l.bias_mu, l.weight_rho, l.bias_rho)],
                           lr=RETRAIN_LR) if arm == "forget_elbo" else None
    env = build_pendulum_env([(0, MASS)], [(0, GRAV)])       # |u|<=2
    torch.manual_seed(seed + 10000)
    obs, _ = env.reset(seed=seed)
    agent.reset(); agent.notify_change()
    pdf = PerDimDriftFilter(n_dims=N_SURP); pdf.reset()

    head = nnhead = aopt = None
    if arm == "head_lin":
        head = LinearActionHead(w0, tau0=1.0, lam_min=1e-3).attach(dyn, sample_w=False)
    elif arm in ("head", "head_noforget"):
        nnhead = NonlinearAdapterHead(w0, obs_dim=3, hid=ADAPT_HID,
                                      prior_std=ADAPT_PRIOR).to(device)
        nnhead.attach(dyn)
        aopt = torch.optim.Adam(nnhead.parameters(), lr=ADAPT_LR)
    agent.alt_dynamics_fn, agent.alt_dynamics_after = (None, 10**9)
    if arm == "oracle":
        def true_dyn(o, a):
            t = torch.atan2(o[:, 1], o[:, 0])
            nth, nwd = analytic(t, o[:, 2], a[:, 0])
            return torch.stack([torch.cos(nth), torch.sin(nth), nwd], -1)
        agent.alt_dynamics_fn, agent.alt_dynamics_after = true_dyn, 0

    buf_in, buf_tgt, perr_buf, rew, perr = [], [], [], [], []
    th_trace, w_trace, dbar_trace, dn_trace, fire_trace = [], [], [], [], []
    n_forget, streak, balanced, min_th = 0, 0, None, math.pi
    for step in range(TRIAL_LEN):
        a = agent.act(obs); aa = np.asarray(a, np.float32).ravel()
        o_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        a_t = torch.as_tensor(aa, dtype=torch.float32, device=device)
        with torch.no_grad():
            zero = torch.zeros_like(a_t.unsqueeze(0))
            m0, lv0 = bnn._run_network(dyn._get_model_input(o_t.unsqueeze(0), zero),
                                       sample=False)
            if head is not None:
                pm, _ = head.predict(bnn, dyn, o_t.unsqueeze(0), a_t.unsqueeze(0))
            elif nnhead is not None:
                pm, _ = nnhead.predict(bnn, dyn, o_t.unsqueeze(0), a_t.unsqueeze(0))
            else:
                pm, _ = bnn._run_network(
                    dyn._get_model_input(o_t.unsqueeze(0), a_t.unsqueeze(0)), sample=False)
        nxt, r, _, _, _ = env.step(a); rew.append(float(r))
        n_t = torch.as_tensor(nxt, dtype=torch.float32, device=device)
        perr.append(float(np.linalg.norm(
            (o_t + pm[0, :3]).cpu().numpy() - np.asarray(nxt, np.float64))))

        th = math.atan2(float(nxt[1]), float(nxt[0])); wv = float(nxt[2])
        th_trace.append(th); w_trace.append(float(head.w[2]) if head is not None
                                            else (float(nnhead.skip.weight_mu[2, 3].detach())
                                                  if nnhead is not None else None))
        min_th = min(min_th, abs(th))
        streak = streak + 1 if (abs(th) <= UP_ANGLE and abs(wv) <= UP_SPEED) else 0
        if streak >= UP_HOLD and balanced is None:
            balanced = step

        if arm in ("forget_elbo", "head_lin", "head", "head_noforget"):
            if head is not None:
                dn, _ = surprise_composed(head, bnn, dyn, o_t, a_t, n_t, sample_w=False)
            elif nnhead is not None:
                with torch.no_grad():
                    ms = torch.stack([nnhead.predict(bnn, dyn, o_t.unsqueeze(0),
                                                     a_t.unsqueeze(0), sample=True)[0][:, :3]
                                      for _ in range(8)])
                    lvs = nnhead.predict(bnn, dyn, o_t.unsqueeze(0),
                                         a_t.unsqueeze(0), sample=False)[1][:, :3]
                    S = ms.var(0, unbiased=False) + torch.exp(lvs) + 1e-12
                    nu = (n_t - o_t)[:3].view(1, -1) - ms.mean(0)
                dn = (nu.pow(2) / S).squeeze(0).cpu().numpy().astype(np.float64)
            else:
                dn = surprise_gaussian(dyn, bnn, obs, aa, nxt, r,
                                       use_reward_dim=False)["delta_n_vec"]
            pdf.update(dn)
            dbar = np.asarray(pdf.delta_bar, dtype=np.float64)[:N_SURP]
            dn_trace.append(float(dn[2])); dbar_trace.append(float(dbar[2]))
            fire_trace.append(float(pdf.drift_estimate()[2]))

        if arm == "head_lin":
            resid = (n_t - o_t).cpu().numpy()[:3] - m0[0, :3].cpu().numpy()
            head.update(float(aa[0]), resid, torch.exp(lv0[0, :3]).cpu().numpy(), dbar)

        if nnhead is not None:
            # replay buffer of raw (obs, act) and full delta-obs targets
            buf_in.append(o_t.unsqueeze(0)); buf_tgt.append(a_t.unsqueeze(0))
            perr_buf.append((n_t - o_t).unsqueeze(0))
            if arm == "head" and (step + 1) % K_FORGET == 0:
                _rho, fired = nnhead.forget(dbar)
                if fired:
                    n_forget += 1
                    nnhead.anchor(include_sigma=True)
            n_buf = len(buf_in)
            idx = (torch.randint(0, n_buf, (ADAPT_BATCH,), device=device)
                   if n_buf > ADAPT_BATCH else torch.arange(n_buf, device=device))
            elbo_step_adapter(nnhead, aopt, bnn, dyn,
                              torch.cat(buf_in)[idx], torch.cat(buf_tgt)[idx],
                              torch.cat(perr_buf)[idx], ACTIVE,
                              n_steps=ADAPT_STEPS, n_mc=ADAPT_MC, beta=BETA,
                              kl_denom=n_buf)

        if arm == "forget_elbo":
            if (step + 1) % K_FORGET == 0:
                _q, fired = forget_gaussian_perdim(bnn, pdf, out_size=OUT_SIZE,
                                                   inflate_mode="retention", q_max=Q_MAX)
                if fired.any():
                    n_forget += 1
                    anchor_head_rows_to_current(bnn, np.where(fired)[0], OUT_SIZE, True)
            with torch.no_grad():
                buf_in.append(dyn._get_model_input(o_t.unsqueeze(0), a_t.unsqueeze(0)))
            buf_tgt.append(torch.cat([n_t - o_t, torch.tensor(
                [r], dtype=torch.float32, device=device)]).unsqueeze(0))
            retrain_gaussian_full_elbo(bnn, opt, torch.cat(buf_in), torch.cat(buf_tgt),
                                       ACTIVE, n_dims_total=OUT_SIZE,
                                       n_steps=RETRAIN_STEPS, n_mc=N_MC, beta=BETA,
                                       kl_denom=len(buf_in), local_reparam=True)
        obs = nxt
    if head is not None:
        w_fin = head.w.tolist(); head.detach()
    elif nnhead is not None:
        w_fin = nnhead.skip.weight_mu[:, 3].detach().cpu().numpy().tolist()
        nnhead.detach()
    else:
        w_fin = None
    return dict(seed=seed, ret=float(np.sum(rew)), pred_err=float(np.mean(perr)),
                min_abs_theta=min_th, balanced=balanced, n_forget=n_forget,
                w_final=w_fin, steps=TRIAL_LEN, per_step_reward=rew,
                dbar_trace=dbar_trace, dn_trace=dn_trace, fire_trace=fire_trace,
                theta_trace=th_trace, w_trace=w_trace,
                ret200=float(np.sum(rew[:200])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--trial-start", type=int, default=0)
    ap.add_argument("--trial-end", type=int, default=None)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out-dir", default="results/pendulum_default_head")
    ap.add_argument("--tag", default="shard")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--mass", type=float, default=MASS,
                    help="deployment mass.  4.0 = the default mass shift; 1.0 = "
                         "unchanged from pretraining.")
    ap.add_argument("--gravity", type=float, default=GRAV,
                    help="deployment gravity.  10.0 = unchanged; 20.0 doubles "
                         "the PASSIVE term, which no action-channel head can "
                         "represent.")
    ap.add_argument("--steps", type=int, default=TRIAL_LEN,
                    help="episode length.  CLAUDE.md's default is 200, but at "
                         "|u|<=2 and mass 4 the control term maxes at "
                         "3u/(m l^2)=1.5 against gravity's 15, so the pole "
                         "cannot be lifted directly and must be pumped up over "
                         "many swings -- 200 steps ends mid-swing-up.")
    A = ap.parse_args()
    globals()["TRIAL_LEN"] = A.steps
    globals()["MASS"] = A.mass
    globals()["GRAV"] = A.gravity
    globals()["W_TARGET"] = 3.0 * DT / (A.mass * 1.0 ** 2)
    OUT = pathlib.Path(A.out_dir); OUT.mkdir(parents=True, exist_ok=True)

    if A.merge:
        rows = {}
        for f in sorted(OUT.glob("shard_*.json")):
            for arm, rs in json.load(open(f)).items():
                rows.setdefault(arm, []).extend(rs)
        json.dump(rows, open(OUT / "merged.json", "w"), indent=1)
        print(f"{'arm':<13}{'n':>5}{'return mean':>13}{'SEM':>8}{'std':>9}"
              f"{'pred_err':>10}{'balanced':>10}{'min|th|':>9}{'w[thdot]':>10}")
        for arm in ARMS:
            rs = rows.get(arm, [])
            if not rs: continue
            r = np.array([x["ret"] for x in rs]); n = len(r)
            w = [x["w_final"][2] for x in rs if x["w_final"]]
            print(f"{arm:<13}{n:>5}{r.mean():>13.1f}{r.std(ddof=1)/math.sqrt(n):>8.1f}"
                  f"{r.std(ddof=1):>9.1f}"
                  f"{np.mean([x['pred_err'] for x in rs]):>10.4f}"
                  f"{np.mean([x['balanced'] is not None for x in rs]):>10.2f}"
                  f"{np.mean([x['min_abs_theta'] for x in rs]):>9.3f}"
                  f"{(f'{np.mean(w):.5f}' if w else '-'):>10}")
        return

    end = A.trial_end if A.trial_end is not None else A.trials
    hid, nl = load_arch(MODEL_DIR)
    bnn, dyn = make_gaussian_bnn(3, 1, hid_size=hid, num_layers=nl)
    dyn.input_normalizer.load(MODEL_DIR); bnn.load(MODEL_DIR, "bnn_dynamics.pth")
    bnn.num_weight_groups = 1; bnn.aleatoric_in_rollout = False
    bnn.anchor_prior_to_current(include_sigma=True)
    init_state = copy.deepcopy(bnn.state_dict())
    w0 = measure_gain(bnn, dyn, gain_probe(), n_dims=3)
    agent = ContinuousCEMAgent(dyn, bnn, 3, 1, device=device, horizon=H,
                               n_cem_iters=CEM_ITERS, n_candidates=CANDIDATES,
                               elite_frac=ELITE, k_models=K_MODELS, cvar_alpha=ALPHA,
                               gamma=GAMMA, obs_project=project_unit_circle,
                               reward_fn=pendulum_reward)
    print(f"DEPLOY: mass {A.mass} (pretrained 1.0), gravity {A.gravity} "
          f"(pretrained 10.0), |u|<={MAX_T}, {TRIAL_LEN} steps\n"
          f"  passive term 3g/2l: {3*10.0/2:.1f} -> {3*A.gravity/2:.1f}   "
          f"control term 3/(m l^2): {3/1.0:.2f} -> {3/A.mass:.2f}\n"
          f"  shard trials [{A.trial_start},{end})  "
          f"w0={np.array2string(w0,precision=5)}  "
          f"true control gain w*=3*dt/(m l^2)={W_TARGET:.4f}", flush=True)
    out, t0 = {}, time.time()
    for arm in A.arms.split(","):
        out[arm] = []
        for i in range(A.trial_start, end):
            out[arm].append(run_trial(arm, SEED_BASE + i, bnn, dyn, init_state,
                                      agent, w0))
            print(f"  {arm:<12} trial {i:3d} seed {SEED_BASE+i} "
                  f"ret={out[arm][-1]['ret']:8.1f} [{time.time()-t0:.0f}s]", flush=True)
        json.dump(out, open(OUT / f"shard_{A.tag}.json", "w"), indent=1)
    print(f"shard done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
