"""Sweep: size of the online adaptation stack x updating the final layer.

All configs load the SAME shipped Dirichlet checkpoint (bnn_dirichlet_k3.pth,
frozen trunk).  For one --n-train-layers N, this script inserts N-1 fresh
identity-init linear layers between the trunk and the Dirichlet head and runs
the run_discrete.py trial loop (slippery p=0.7 from step 0, surprise ->
DriftFilterV2 -> forget every step, conjugate counts on) TWICE:

  head1 : the adapters AND the Dirichlet head are gradient-retrained on the
          post-change buffer every K_FORGET steps ("with final-layer update");
  head0 : only the adapters train, the head stays frozen ("without update").
          At N=1 this is the original pipeline (nothing gradient-trains).

One JSON per variant with per-trial returns/steps/outcomes.

SURPRISE LOGGING.  Every trial line now also reports the surprise/detection
state (delta_n mean/median/max, the step and dbar of the FIRST detection, final
and mean dbar, how many steps forget actually fired, final retain), each variant
ends with an across-trial summary (mean +/- 95% CI), and the same aggregates are
stored under `surprise` in the JSON alongside the per-trial fields.  None of it
touches the trajectory -- these are read-outs of state the loop already computes,
so the numbers are directly comparable to runs made before this was added.
`--verbose-steps K` additionally prints the full per-step trace for the first K
trials (as in run_discrete.py), and `--trace-arrays` keeps the per-step arrays in
the JSON.

Run (one process per N):
    conda activate nsgym
    python sweep_dirichlet_layers.py --n-train-layers 3 --trials 40 \
        --out-dir results/adapter_sweep
    # g3-style alpha cell with a full trace for the first 2 trials:
    python sweep_dirichlet_layers.py --n-train-layers 1 --head 0 --cvar-alpha 0.5 \
        --max-steps 500 --trials 40 --out-dir results/g3 --verbose-steps 2
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
    make_dirichlet_bnn, surprise_dirichlet, epistemic_dirichlet, forget_dirichlet,
    mean_alpha0, direction_of, unfrozen_params_dirichlet, retrain_dirichlet,
)
from planning.cvar_cem import CVaRCEMAgent

ACTS = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
DIRS = {0: "intend", 1: "perp-", 2: "perp+", -1: "none"}


_VAR_GRID = {}


def predictive_variance(bnn, per_pair_out=False):
    """Mean total Dirichlet variance sum_i Var(p_i) over ALL 16x4 (s,a).

    Var(p_i) = a_i (a0 - a_i) / (a0^2 (a0 + 1)) on the CURRENT effective alpha, so
    it includes BOTH online channels -- forget's retain shrink (which inflates it
    toward the CONC_PRIOR ceiling) and the conjugate counts (which contract it) --
    plus whatever the gradient retrain has done to the weights.  One batched
    mean-weight forward per call, so it is cheap enough to log every step.
    """
    key = (bnn.n_states, bnn.n_actions)
    if key not in _VAR_GRID:
        rows = []
        for s in range(bnn.n_states):
            for a in range(bnn.n_actions):
                x = np.zeros(bnn.n_states + bnn.n_actions, dtype=np.float32)
                x[s] = 1.0; x[bnn.n_states + a] = 1.0
                rows.append(x)
        _VAR_GRID[key] = torch.as_tensor(np.stack(rows), device=device)
    with torch.no_grad():
        alpha, _, _, _ = bnn._forward_alpha(_VAR_GRID[key], sample=False,
                                            num_weight_groups=1)
        a0 = alpha.sum(-1, keepdim=True)
        var = alpha * (a0 - alpha) / (a0 ** 2 * (a0 + 1.0))
        per_pair = var.sum(-1)                       # (64,) rows are s*4+a
        if per_pair_out:
            return float(per_pair.mean().item()), per_pair.cpu().numpy()
        return float(per_pair.mean().item())


def unfrozen_layers(bnn, n_unfrozen):
    """The layer OBJECTS unfrozen_params_dirichlet draws its parameters from,
    same top-down ordering: [dir_head, trunk_top, ..., trunk_bottom]."""
    path = [bnn.bayes_layers[bnn.n_trunk]]
    path += list(reversed(list(bnn.bayes_layers[:bnn.n_trunk])))
    return path[:max(0, int(n_unfrozen))]


def variational_params(layers):
    """mu AND rho of the given layers -- the parameter set whose update moves the
    posterior MEAN and its WIDTH.  (unfrozen_params_dirichlet returns mu only.)"""
    params = []
    for layer in layers:
        params += [layer.weight_mu, layer.bias_mu, layer.weight_rho, layer.bias_rho]
    return params


def mean_weight_sigma(layers):
    """Mean posterior width over the given layers -- the scalar that says whether
    the last update inflated or contracted the belief."""
    if not layers:
        return float("nan")
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for layer in layers:
            for sig in (layer.weight_sigma, layer.bias_sigma):
                tot += float(sig.sum().item()); cnt += sig.numel()
    return tot / max(cnt, 1)


def anchor_layers(layers):
    """Re-anchor the KL prior (mean AND width) to the CURRENT belief.  Called
    right after a forget inflation so the following ELBO contracts from the
    inflated posterior instead of from N(., prior_std=1)."""
    for layer in layers:
        layer.anchor_prior_to_current(include_sigma=True)


def retrain_dirichlet_elbo(bnn, opt, layers, model_in, s2_idx, n_steps=5):
    """ELBO retrain that updates BOTH the posterior mean and its width.

    The shipped retrain_dirichlet scores the MEAN-weight forward (sample=False),
    so weight_rho/bias_rho carry no gradient and only mu can move.  Here the NLL
    is Monte-Carlo'd over SAMPLED weights (reparameterization), so sigma is in the
    graph, and a KL against the anchored prior keeps the width from collapsing:

        loss = E_w[ -log p(s2 | w) ] + beta * KL(q || anchored prior) / B

    KL is summed over the retrained layers only.  Returns (nll, kl, loss).
    """
    B = model_in.shape[0]
    nll = kl = loss = None
    for _ in range(n_steps):
        opt.zero_grad()
        nll_acc = torch.zeros(1, device=bnn.device)
        for _ in range(bnn.num_mc_samples):
            mean, _ = bnn._run_network(model_in, sample=True)
            p = mean[:, :bnn.n_states].clamp_min(1e-12)
            nll_acc = nll_acc + (-torch.log(p.gather(1, s2_idx)).mean())
        nll = nll_acc / bnn.num_mc_samples
        kl = sum(layer.kl_divergence() for layer in layers)
        loss = nll + bnn.beta * kl / B
        loss.backward()
        opt.step()
    return float(nll.item()), float(kl.item()), float(loss.item())

SCHEDULE = [(0, 0.7)]     # slippery from step 0 (the "change"), as in run_discrete
CHANGE_STEP = 0
TRIAL_LEN = 100
K_FORGET = 1
RETRAIN_LR = 1e-3
RETRAIN_STEPS = 5
N_ACTIONS = 4
GAMMA = 0.99              # discount for the reported expected discounted return


def run_trials(bnn, dyn, env, init_state, update_final, n_trials, seed_base, log,
               no_forget=False, cvar_alpha=1.0, trial_len=TRIAL_LEN,
               verbose_trials=0, step_every=1, trace_arrays=False,
               n_train_layers=1, max_forgets=None, retrain_variance=False):
    """The run_discrete.py loop with online adapter retraining.

    cvar_alpha : CVaR tail fraction for the risk-averse planner.  1.0 = full
    distribution = risk-neutral; lower (toward 0.05) scores each candidate by an
    ever-worse tail, so the planner reacts more strongly to the epistemic
    uncertainty that forget re-inflates.  adaptive_alpha stays off (fixed alpha).
    trial_len  : hard per-trial step cap (safety).  With the env's own TimeLimit
    raised, set this large so an episode ends only on goal/hole, not truncation.

    Surprise instrumentation (does not change the trajectory -- the extra
    quantities are read-outs of state the loop already computes):
    verbose_trials : print the full per-step surprise/drift/forget trace for the
        first this-many trials (0 = off, the sweep's original quiet behaviour).
        Costs one extra epistemic_dirichlet read-out per LOGGED step only.
    step_every     : thin the per-step trace to every Nth step.
    trace_arrays   : also store the per-step arrays (nll, entropy, dbar, retain,
        rho, p_reached, alpha0) in the JSON, not just the per-trial summary.
    """
    # The inserted-adapter stack this script was written against is gone; the bnn
    # package now adapts by unfreezing the top layers of the DIRECTION path in
    # place (unfrozen_params_dirichlet).  Map the old knob onto the new one:
    #   head1 -> head + the N-1 layers below it;  head0 -> those N-1 layers only.
    # At N=1 this reproduces the original meaning exactly (head0 = fully frozen,
    # nothing gradient-trains -- the configuration behind results/g3).
    def n_unfrozen(update_head):
        return n_train_layers if update_head else n_train_layers - 1

    bnn.num_weight_groups = 20
    bnn.use_counts = True
    rng = np.random.default_rng(seed=0)
    agent = CVaRCEMAgent(dyn, bnn, env.env.desc, device,
                         n_actions=N_ACTIONS, rng=rng,
                         cvar_alpha=cvar_alpha, adaptive_alpha=False, gamma=GAMMA)
    records = []
    for trial in range(n_trials):
        torch.manual_seed(seed_base + 10000 + trial)
        obs, _ = env.reset(seed=seed_base + trial)
        agent.reset()
        bnn.load_state_dict(init_state)  # pretrained weights + identity adapters
        bnn.retain.fill_(1.0)
        bnn.reset_counts()
        train_layers = unfrozen_layers(bnn, n_unfrozen(update_final))
        retrain_params = (variational_params(train_layers) if retrain_variance
                          else unfrozen_params_dirichlet(bnn, n_unfrozen(update_final)))
        retrain_opt = (torch.optim.Adam(retrain_params, lr=RETRAIN_LR)
                       if retrain_params else None)
        if retrain_variance:
            # start from the pretrained belief as the KL anchor; each forget
            # re-anchors to the freshly inflated one
            anchor_layers(train_layers)
        buf_X, buf_s2 = [], []
        drift = DriftFilterV2(eta=ETA, gamma_uncertainty=GAMMA_UNCERTAINTY)
        terminated = truncated = False
        steps, post, total_return, reward = 0, 0, 0.0, 0.0
        disc_ret, deltas = 0.0, []
        # ── surprise / drift / forget instrumentation for this trial ────────────
        verbose = trial < verbose_trials
        nlls, entropies, p_reacheds, alpha0s = [], [], [], []
        dbars, retains, rhos, pvars = [], [], [], []
        pvar_pairs = []          # per-step vector of the 64 per-(s,a) variances
        retain_steps = []        # retain AFTER each step (forget trajectory)
        dir_counts = {0: 0, 1: 0, 2: 0, -1: 0}
        first_detect_step, first_detect_dbar = None, None
        n_forget_fired = 0
        if verbose:
            log(f"  ---- trial {trial+1} per-step trace "
                f"(head={int(update_final)} alpha={cvar_alpha:g}) ----")
        t0 = time.time()
        while not (terminated or truncated) and steps < trial_len:
            if steps == CHANGE_STEP:
                drift.reset()
                agent.notify_change()
            agent.surprise_bar = drift.delta_bar
            agent.n_since_change = post
            action = agent.act(obs)
            a = int(np.argmax(action))
            one_hot_action = to_one_hot_action(action)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_return += max(0.0, float(reward))
            disc_ret += (GAMMA ** steps) * float(reward)
            s = int(np.argmax(obs)); s2 = int(np.argmax(next_obs))
            d = direction_of(s, a, s2)
            vs = surprise_dirichlet(dyn, bnn, obs, one_hot_action, next_obs, reward)
            deltas.append(round(vs["delta_n"], 4))
            # record the surprise decomposition BEFORE the count/forget updates,
            # i.e. the true one-step predictive quantities
            nlls.append(round(vs["nll"], 4))
            entropies.append(round(vs["entropy"], 4))
            p_reacheds.append(round(vs["p_reached"], 5))
            alpha0s.append(round(vs["alpha0"], 3))
            dir_counts[int(d)] = dir_counts.get(int(d), 0) + 1
            excess = vs["delta_n"] - drift.baseline
            drift.update(vs["delta_n"])
            dbars.append(round(drift.delta_bar, 4))
            if first_detect_dbar is None and drift.delta_bar > 1.0:
                first_detect_step, first_detect_dbar = steps, float(drift.delta_bar)
            do_log = verbose and steps % step_every == 0
            bnn.add_count(s, a, d)
            if do_log:
                pd = vs["p_dir"]
                eps = epistemic_dirichlet(dyn, bnn, obs, one_hot_action)
                n_sa = int(bnn.counts[s, a].sum().item())   # incl. this transition
                log(f"   t={steps:3d} | {s:2d},{ACTS[a]:5s}->{s2:2d} "
                    f"dir={DIRS.get(int(d), '?'):6s} p(s'|model)={vs['p_reached']:5.3f}"
                    f" || delta_n={vs['delta_n']:7.3f} nll={vs['nll']:6.3f} "
                    f"H={vs['entropy']:5.3f} alpha0={vs['alpha0']:7.3f} "
                    f"epi={eps['epistemic']:.5f} p_dir=[{pd[0]:.2f},{pd[1]:.2f},{pd[2]:.2f}]"
                    f" || n(s,a)={n_sa:2d} retain={float(bnn.retain.item()):.4e}"
                    f" || excess={excess:+7.3f} lam={drift.lambda_hat:+7.3f}"
                    f"(+/-{drift.lambda_sd:.3f}, n={drift._n:3d}) "
                    f"base={drift.baseline:5.3f} dbar={drift.delta_bar:7.3f}")
            if retrain_opt is not None:
                buf_X.append(np.concatenate([obs, one_hot_action]).astype(np.float32))
                buf_s2.append(s2)
            post += 1
            if post % K_FORGET == 0:
                # max_forgets caps the number of ACTUAL firings (rho<1); a
                # deadband no-op inflates nothing and does not count.
                may_forget = (not no_forget and
                              (max_forgets is None or n_forget_fired < max_forgets))
                if may_forget:
                    # mean_alpha0 sweeps all 16x4 (s,a) -- only pay for it on a
                    # step whose trace is actually printed.
                    a0_before = mean_alpha0(dyn, bnn) if do_log else None
                    rho, ret_b, ret_a = forget_dirichlet(bnn, drift)
                    rhos.append(round(rho, 5))
                    if rho < 1.0:
                        n_forget_fired += 1
                        if retrain_variance:
                            # re-anchor so the ELBO contracts FROM the inflated
                            # belief, not from the fixed prior_std
                            anchor_layers(train_layers)
                    if do_log:
                        a0_after = mean_alpha0(dyn, bnn)
                        log(f"        >>> FORGET @ t={steps}: dbar={drift.delta_bar:.3f} "
                            f"drift_est={drift.drift_estimate():+.3f} -> rho={rho:.4f}"
                            f" ({'fired' if rho < 1.0 else 'no-op, dbar<=1'})"
                            f" | retain {ret_b:.4e}->{ret_a:.4e}"
                            f" | mean alpha0 {a0_before:.3f}->{a0_after:.3f}")
                retains.append(float(bnn.retain.item()))
                if retrain_opt is not None and buf_X:
                    X = torch.as_tensor(np.stack(buf_X), device=device)
                    s2_idx = torch.as_tensor(
                        buf_s2, dtype=torch.long, device=device).unsqueeze(1)
                    if retrain_variance:
                        nll_r, kl_r, _ = retrain_dirichlet_elbo(
                            bnn, retrain_opt, train_layers, X, s2_idx,
                            n_steps=RETRAIN_STEPS)
                        if do_log:
                            log(f"        >>> RETRAIN(ELBO mu+sigma) @ t={steps}: "
                                f"nll={nll_r:+.4f} kl={kl_r:.1f} buf={len(buf_X)} "
                                f"| mean sigma={mean_weight_sigma(train_layers):.5f}")
                    else:
                        retrain_dirichlet(bnn, retrain_opt, X, s2_idx,
                                          n_steps=RETRAIN_STEPS)
            # predictive variance AFTER this step's forget / counts / retrain --
            # i.e. the model state the next decision will be made with
            retain_steps.append(round(float(bnn.retain.item()), 8))
            _pv_mean, _pv_vec = predictive_variance(bnn, per_pair_out=True)
            pvars.append(round(_pv_mean, 6))
            if trace_arrays:
                pvar_pairs.append([round(float(v), 5) for v in _pv_vec])
            obs = next_obs
            steps += 1
        outcome = ("goal" if reward > 0 else "hole") if terminated else "truncated"
        dn = np.asarray(deltas, dtype=float)
        rec = dict(ret=total_return, disc_ret=round(disc_ret, 5),
                   steps=steps, outcome=outcome, deltas=deltas,
                   # ── surprise / drift / forget summary for this trial ─────────
                   dn_mean=round(float(dn.mean()), 4),
                   dn_median=round(float(np.median(dn)), 4),
                   dn_max=round(float(dn.max()), 4),
                   nll_mean=round(float(np.mean(nlls)), 4),
                   entropy_mean=round(float(np.mean(entropies)), 4),
                   p_reached_mean=round(float(np.mean(p_reacheds)), 5),
                   alpha0_first=alpha0s[0], alpha0_last=alpha0s[-1],
                   first_detect_step=first_detect_step,
                   first_detect_dbar=(round(first_detect_dbar, 4)
                                      if first_detect_dbar is not None else None),
                   dbar_final=round(float(drift.delta_bar), 4),
                   dbar_mean=round(float(np.mean(dbars)), 4),
                   lambda_sd_final=round(float(drift.lambda_sd), 5),
                   baseline=round(float(drift.baseline), 4),
                   retain_final=float(bnn.retain.item()),
                   rho_min=(round(min(rhos), 5) if rhos else None),
                   n_forget_fired=n_forget_fired,
                   sigma_final=round(mean_weight_sigma(train_layers), 6),
                   # EXACT visit distribution: bnn.counts[s,a,d] is the conjugate
                   # tally, so summing the direction axis gives visits per (s,a).
                   # Free -- the loop already maintains it.
                   visit_counts=[[int(v) for v in row]
                                 for row in bnn.counts.sum(-1).cpu().numpy().tolist()],
                   n_pairs_visited=int((bnn.counts.sum(-1) > 0).sum().item()),
                   n_states_visited=int((bnn.counts.sum(-1).sum(-1) > 0).sum().item()),
                   pvar_first=pvars[0], pvar_final=pvars[-1],
                   pvar_max=max(pvars),
                   # dbar at the FIRST forget firing == the surprise level that
                   # actually triggered inflation (what the plot second panel shows)
                   dbar_at_first_forget=(round(dbars[first_detect_step], 4)
                                         if first_detect_step is not None else None),
                   dir_counts={DIRS.get(k, str(k)): v for k, v in dir_counts.items()})
        if trace_arrays:
            rec.update(nlls=nlls, entropies=entropies, p_reached=p_reacheds,
                       alpha0s=alpha0s, dbars=dbars, rhos=rhos, pvars=pvars,
                       pvar_pairs=pvar_pairs, retain_steps=retain_steps,
                       retains=[float(f"{r:.6e}") for r in retains])
        records.append(rec)
        _fd = (f"t={first_detect_step} dbar={first_detect_dbar:.3f}"
               if first_detect_dbar is not None else "never")
        log(f"  head={int(update_final)} forget={int(not no_forget)} "
            f"trial {trial+1}/{n_trials}: {outcome} steps={steps} "
            f"disc={disc_ret:+.3f} | dn mean={dn.mean():6.3f} med={np.median(dn):6.3f} "
            f"max={dn.max():7.3f} | first_detect[{_fd}] | dbar final={drift.delta_bar:6.3f} "
            f"mean={np.mean(dbars):6.3f} | forget fired {n_forget_fired}/{len(rhos)} "
            f"| retain={float(bnn.retain.item()):.3e} "
            f"| sigma={mean_weight_sigma(train_layers):.5f} ({time.time()-t0:.0f}s)")
    return records


def _mean_or_none(vals):
    """Mean over the non-None entries; None when there are none (never NaN, which
    json.dumps would emit as the non-standard literal NaN)."""
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def log_surprise_summary(records, log):
    """Across-trial surprise/detection aggregates (mean +/- 95% CI over trials).

    The dbar-at-first-detection line is the number the FrozenLake default asks
    for: how surprised the filter was the first time it flagged the change.
    """
    def ci95(v):
        v = np.asarray([x for x in v if x is not None], dtype=float)
        if len(v) == 0:
            return float("nan"), float("nan"), 0
        if len(v) == 1:
            return float(v[0]), 0.0, 1
        return float(v.mean()), 1.96 * float(v.std(ddof=1) / np.sqrt(len(v))), len(v)

    det = [r["first_detect_step"] for r in records]
    n_det = sum(x is not None for x in det)
    m_t, e_t, _ = ci95(det)
    m_d, e_d, _ = ci95([r["first_detect_dbar"] for r in records])
    m_dn, e_dn, _ = ci95([r["dn_mean"] for r in records])
    m_df, e_df, _ = ci95([r["dbar_final"] for r in records])
    m_p, e_p, _ = ci95([r["p_reached_mean"] for r in records])
    m_h, e_h, _ = ci95([r["entropy_mean"] for r in records])
    log("  ---- surprise summary over %d trials ----" % len(records))
    log(f"    detected change in {n_det}/{len(records)} trials | "
        f"first detect at t={m_t:.2f}+/-{e_t:.2f} with dbar={m_d:.3f}+/-{e_d:.3f}")
    log(f"    delta_n per trial: mean={m_dn:.3f}+/-{e_dn:.3f} | "
        f"p(realized s'|model)={m_p:.4f}+/-{e_p:.4f} | entropy={m_h:.3f}+/-{e_h:.3f}")
    m_s, e_s, _ = ci95([r.get("sigma_final") for r in records])
    log(f"    dbar at trial end={m_df:.3f}+/-{e_df:.3f} | "
        f"forget fired {np.mean([r['n_forget_fired'] for r in records]):.1f} steps/trial | "
        f"final retain (median)={np.median([r['retain_final'] for r in records]):.3e} | "
        f"final mean posterior sigma={m_s:.5f}+/-{e_s:.5f}")
    tot = {}
    for r in records:
        for k, v in r["dir_counts"].items():
            tot[k] = tot.get(k, 0) + v
    n_all = max(1, sum(tot.values()))
    log("    realized directions: "
        + " ".join(f"{k}={v} ({100*v/n_all:.1f}%)" for k, v in tot.items() if v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train-layers", type=int, required=True,
                    help="top linear layers in the adaptation stack, incl. the "
                         "Dirichlet head; N-1 adapters are inserted")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="shift per-trial seeds by this so a follow-up run adds "
                         "DISTINCT trials (e.g. 60 more with --seed-offset 40)")
    ap.add_argument("--out-dir", default="results/adapter_sweep")
    ap.add_argument("--head", choices=["0", "1", "both"], default="both",
                    help="which final-layer variant(s) to run")
    ap.add_argument("--no-forget", action="store_true",
                    help="skip forget_dirichlet entirely (ablation)")
    ap.add_argument("--cvar-alpha", type=float, default=1.0,
                    help="CVaR tail fraction (1.0 risk-neutral, lower risk-averse)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="raise the env TimeLimit AND per-trial cap to this, so "
                         "episodes end on goal/hole, not truncation (default: 100)")
    ap.add_argument("--verbose-steps", type=int, default=0, metavar="K",
                    help="print the full per-step surprise/drift/forget trace for "
                         "the first K trials (0 = off; the trajectory is unchanged, "
                         "these are read-outs of state the loop already computes)")
    ap.add_argument("--step-every", type=int, default=1, metavar="N",
                    help="with --verbose-steps, print only every Nth step")
    ap.add_argument("--env-p", type=float, default=0.7,
                    help="intended-prob the ENV runs at from t=0 (the post-change "
                         "dynamics). The model's belief comes from --ckpt.")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint filename under data/frozenlake to load as the "
                         "pretrained belief (default: the shipped deterministic one)")
    ap.add_argument("--max-forgets", type=int, default=None, metavar="K",
                    help="cap the number of ACTUAL forget firings (rho<1) per "
                         "trial; deadband no-ops don't count (default: no cap). "
                         "K=2 is the saturation point measured on the g3 data")
    ap.add_argument("--retrain-variance", action="store_true",
                    help="retrain BOTH the posterior mean and its width: ELBO "
                         "(MC-sampled NLL + KL vs the anchored prior) over "
                         "mu AND rho, re-anchored after each forget. Without it "
                         "the shipped mean-weight NLL moves only mu")
    ap.add_argument("--trace-arrays", action="store_true",
                    help="store the per-step surprise arrays (nll, entropy, dbar, "
                         "retain, rho, p_reached, alpha0) in the JSON, not just the "
                         "per-trial summary scalars")
    args = ap.parse_args()

    trial_len = args.max_steps if args.max_steps is not None else TRIAL_LEN

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # tag includes alpha (and 'notrunc' when the cap is raised) so runs don't clash.
    astr = ("a" + f"{args.cvar_alpha:g}".replace(".", "p")) if args.cvar_alpha != 1.0 else ""
    tstr = "_notrunc" if args.max_steps is not None else ""
    fstr = f"_f{args.max_forgets}" if args.max_forgets is not None else ""
    vstr = "_rv" if args.retrain_variance else ""
    tag = f"N{args.n_train_layers}{astr}{tstr}{fstr}{vstr}"

    def log(*a):
        print(f"[{tag}]", *a, flush=True)

    torch.manual_seed(args.seed + args.n_train_layers)
    np.random.seed(args.seed + args.n_train_layers)

    sched = [(0, args.env_p)]
    env = build_scheduled_env(sched, max_episode_steps=args.max_steps)
    bnn, dyn = make_dirichlet_bnn(16, 4)
    if args.ckpt:
        bnn.load(SAVE_DIR, filename=args.ckpt)
    else:
        bnn.load(SAVE_DIR)              # shared shipped checkpoint
    init_state = copy.deepcopy(bnn.state_dict())
    log(f"loaded {args.ckpt or 'shipped bnn_dirichlet_k3.pth'}; env p={args.env_p} "
        f"-> {[round(x, 3) for x in (args.env_p, (1-args.env_p)/2, (1-args.env_p)/2)]}; "
        f"n_train_layers={args.n_train_layers} "
        f"(unfrozen direction-path layers: head1={args.n_train_layers}, "
        f"head0={args.n_train_layers - 1}) trial_len={trial_len}")

    variants = {"0": (False,), "1": (True,), "both": (True, False)}[args.head]
    for update_final in variants:
        records = run_trials(bnn, dyn, env, init_state, update_final,
                             args.trials, seed_base=args.seed + 1000 + args.seed_offset,
                             log=log, no_forget=args.no_forget,
                             cvar_alpha=args.cvar_alpha, trial_len=trial_len,
                             verbose_trials=args.verbose_steps,
                             step_every=max(1, args.step_every),
                             trace_arrays=args.trace_arrays,
                             n_train_layers=args.n_train_layers,
                             max_forgets=args.max_forgets,
                             retrain_variance=args.retrain_variance)
        log_surprise_summary(records, log)
        rets = [r["ret"] for r in records]
        disc = [r["disc_ret"] for r in records]
        out = dict(
            n_train_layers=args.n_train_layers, update_final=update_final,
            forget=not args.no_forget, cvar_alpha=args.cvar_alpha, gamma=GAMMA,
            trials=args.trials, seed=args.seed, schedule=sched,
            trial_len=trial_len, k_forget=K_FORGET,
            env_p=args.env_p, ckpt=args.ckpt,
            max_forgets=args.max_forgets, retrain_variance=args.retrain_variance,
            retrain_lr=RETRAIN_LR, retrain_steps=RETRAIN_STEPS,
            mean_return=float(np.mean(rets)),
            mean_disc_return=float(np.mean(disc)),
            # ── surprise aggregates (mirrors the printed summary) ───────────────
            surprise=dict(
                n_detected=int(sum(r["first_detect_step"] is not None for r in records)),
                # None (not NaN) when no trial ever detected -- keeps the JSON strict
                mean_first_detect_step=_mean_or_none(
                    [r["first_detect_step"] for r in records]),
                mean_first_detect_dbar=_mean_or_none(
                    [r["first_detect_dbar"] for r in records]),
                mean_dn=float(np.mean([r["dn_mean"] for r in records])),
                mean_dbar_final=float(np.mean([r["dbar_final"] for r in records])),
                mean_p_reached=float(np.mean([r["p_reached_mean"] for r in records])),
                mean_entropy=float(np.mean([r["entropy_mean"] for r in records])),
                mean_forget_fired=float(np.mean([r["n_forget_fired"] for r in records])),
                median_retain_final=float(np.median([r["retain_final"] for r in records])),
            ),
            records=records,
        )
        suffix = "_noforget" if args.no_forget else ""
        path = out_dir / f"{tag}_head{int(update_final)}{suffix}.json"
        path.write_text(json.dumps(out, indent=2))
        _mfd = out["surprise"]["mean_first_detect_dbar"]
        log(f"saved {path}: disc return {np.mean(disc):+.3f} "
            f"goal rate {np.mean([r['outcome'] == 'goal' for r in records]):.3f} "
            f"| mean delta_n {out['surprise']['mean_dn']:.3f} "
            f"| mean dbar@detect {'n/a' if _mfd is None else f'{_mfd:.3f}'}")


if __name__ == "__main__":
    main()
