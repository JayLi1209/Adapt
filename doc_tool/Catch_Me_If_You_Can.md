# Online Risk-Averse Method for Control in Non-Stationary Environments

Yuanhe Li William & Mary

Chenan Wang William & Mary

Haipeng Chen William & Mary

Ayan Mukhopadhyay William & Mary

## Abstract

Current model-based reinforcement learning algorithms lack a mechanism to adapt to informed but unbounded changes safely. We present forget-inflate-retrain (FIR), a mechanism to make Bayesian systems adapting to shifts efficiently and safely. To our knowledge, it is the first algorithm to adapt to environmental changes at the modeling level for non-stationary MDPs. Using Bayesian Neural Network (BNN), FIR chases announced changes through an open loop of four operations: a calibrated surprise score detects how wrong the old model is; a drift filter smooths the noisy surprise stream; forgetting re-inflates the model uncertainty in proportion to the measured drift (retracting the Dirichlet concentrations or the variational precisions); and learning rebuilds the model from fresh post-change evidence (conjugate online counts, or ELBO retraining once enough data is gathered). On top of this model, a CEM planner with a CVaR risk criterion makes risk-averse decisions that are confidence-gated: the CVaR tail fraction is driven by the same drift signal, so the agent is maximally cautious right after a change and relaxes as its belief recovers. We demonstrate its accuracy, efficiency, and safety against established baselines from the NSMDP community (RATS, ADA-MCTS, DP-snapshot, DP-NSMDP). Code can be found at https://github.com/JayLi1209/Adapt.

## 1 Introduction

Adapting to an unexpected change is a long-standing problem in disaster responses, autonomous driving and possibly more. [possible citations to older Ayan, Ma, or Liakot’s papers]. In such settings, unexpected changes in environment can happen suddenly in a great magnitude. Key difficulties in designing such an adaptive algorithm come into realizing these two objectives: 1) efficiently absorb the initial impact of this change, and 2) start to cautiously explore the new environment after change. In this paper, we assume the problems are framed in nonstationary Markov Decision Processes (NSMDPs; Lecar pentier paper). Developing and adapting to these two objectives in a timely manner remains an open question.

One popular paradigm in solving the problem is to train a base model offline and then adapt to changes online. These model-based reinforcement learning approaches are widely employed [ADA-MCTS paper, RATS paper], because it is heuristic and flexible. Using a model offers many benefits over model-free reinforce ment learning. First, it makes more efficient use of prior data to learn a better schedule. Moreover, it could can be independent of any specific task and thus have the potential to transfer well to other tasks in the environment. [PlaNet paper]

During model training, Bayesian systems give extra uncertainty information during modeling, and helps with the planning as seen in [ADA-MCTS paper]. By inherently propagating a variance noise throughout learning, Bayesian approaches give a statistical measure of the “belief” strength and hence as a training signal of under explored terrains. It is reasonable to use Bayesian Neural Network (BNN).

We propose FIR, a Bayesian training framework aiming to adapt to the post-change environment safely. When an informed environmental shift happens, FIR does the following:

1. Forgetting. It calculates the “distance” of the observed transition to the old posterior predictive distribution and produces a calibrated surprise score, the quantity that later drives every update.

2. Smoothing. It smooths out the noise of this singular measurement by maintaining an equal-weight moving-average drift filter with an empirically calibrated baseline.

3. Inflating. The smoothed drift magnitude is incorporated into inflating the variance of the BNN prediction: the Dirichlet head's concentrations are retracted toward a symmetric prior (discrete states), or the variational precisions are relaxed toward / beyond the prior (continuous states). This destroys the stale confidence that produced “confident but wrong” predictions.

4. Retraining. After a certain amount of fresh data is gathered, the BNN is calibrated with ELBO updates on that data; meanwhile, cheap conjugate updates (online Dirichlet counts) keep the head tracking the new environment. Forget stops firing once the surprise is back to baseline, and retraining/learning takes the lead.

On top of this model, the agent plans with CEM + CVaR: cross-entropy method action-sequence search whose score is the empirical Conditional Value-at-Risk of the imagined returns (over posterior model draws × aleatoric rollouts), with the CVaR tail fraction α driven by the same confidence signal. This gives a principled risk-aversion knob: maximally cautious (small α) when the belief is stale, relaxing to risk-neutral (α→1) as the model recovers.

Our contributions are:

1. FIR, the first model-level BNN surgery mechanism to accommodate change: surprise → drift → forget (inflate) → learn/retrain, with a proven sample-complexity bound and no retraining required for safe recovery in the discrete case.

2. A confidence-gated CVaR-CEM planner combining a Bayesian world model with a risk-averse MPC objective, where the same drift signal controls both the model uncertainty and the risk criterion.

3. Strong empirical performance against SOTA baselines on the NSMDP community benchmarks (FrozenLake, CliffWalking, NS-Bridge, Pendulum).

## 2 Preliminaries

After an environmental change, safe planning includes modeling the uncertainty, and makeing risk-averse decisions.

## 2.1 Problem Assumptions

We consider a non-stationary Markov Decision Process (NSMDP) denoted as $\mathcal { M } _ { t } = ( { \mathcal { S } } , { \mathcal { A } } , P _ { t } , r _ { t } , \gamma )$ , where S,A denotes the state and action space, $P _ { t } \left( s ^ { \prime } | s , a \right)$ is the transition probability, $r _ { t } ( s , a )$ is the reward function, and γ is the discount factor. We assume environmental changes can only happen in discrete points in time, hence the changes can be completely characterized by $P _ { t }$ and $r _ { t } .$ two functions unknown to the planning agent. We also assume that the agent observes one trajectory and cannot reset the environment to a previous $( s , a )$ to re-explore other actions, befitting the definition of “online" algo rithm.

A common strategy to plan out the change is to use Model Predictive Control (MPC), where before deciding on an action, the agent takes a snapshot of the environ ment and do planning on its simulated snapshot environ ment; action selection is hence the argmax of the plans that has the largest monte-carlo return.

Two properties of the setting make adaptation hard but tractable: (1) the change is *informed* — the agent knows a change has happened (e.g. via an anomaly detector) but not what it is, so it must act before it can re-learn; (2) the change is *unbounded* — we make no Lipschitz assumption on the transition evolution (unlike the NSMDP worst-case line of work), so prior data can become arbitrarily wrong and must be actively de-weighted rather than merely discounted.

## 2.2 Modeling Changes

We model the (unknown) environment with a Bayesian Neural Network (BNN): variational inference places a Gaussian posterior over the weights, and predictions are made by sampling networks from that posterior. This gives a principled decomposition of uncertainty [Kendall & Gal, ICML 2018]: *aleatoric* uncertainty — the irreducible stochasticity of the environment — is captured by the entropy of the predictive distribution (for a categorical head, the Dirichlet mean; for a Gaussian head, the predictive variance), while *epistemic* uncertainty — the model's lack of knowledge — is captured by the spread of predictions across posterior weight draws. Epistemic uncertainty is exactly the quantity a change destroys: after an environmental shift, the old posterior is *confidently wrong*, so the epistemic estimate must be re-inflated, not merely trusted. This motivates modeling the head as a Dirichlet-Multinomial posterior (discrete states), which enjoys exact conjugate updates: the predictive mean and the concentration are both closed-form functions of the retained prior and the observed counts, which is what makes FIR's forget-and-learn loop cheap and interpretable. For continuous states, a Gaussian head with a variational posterior plays the same role, with the re-inflation acting on the weight precisions.

## 2.3 Adapting to Changes

Many classes of algorithms aim to address environmental shifts through different problem modelings.

Meta learning approaches learn a generalization of previous environments, and linearly combine them to zero-shot new environments. They achieve it by com pressing into a low-dimensional function space [1], a VAE latent [2], an GP [3] or BNN [4] that umbrellas over the entire all change dynamics, or folded into the environment as hidden parameters (HiP-MDP). However, the exogeneous changes in NS-MDP makes them hard to extrapolate. Plus, it is meaningless to learn such “world model" because of such un-genralizable changes.

Domain adaptation approaches align a fixed source to a fixed target, often by training an adversary to strip away domain-specific signal and leave an invariant representation. Non-stationarity offers neither a fixed target nor a privileged invariant. Examples: GAN and its variants. Also, it stuggles when non-stationarity is structural changes instead of parameter drifts.

## 3 Method

The method has two stages: an offline pretraining stage that builds a tight, goal-informed belief over the *original* environment, and an online stage (FIR) that detects, absorbs, and learns from an announced change while a risk-aware planner keeps the agent safe throughout. Figure 1 overviews the online loop.

## 3.1 Offline Training

We pretrain the Bayesian dynamics model on the original environment (e.g. FrozenLake with slip p = 0.7) with two design choices:

1. **Goal-conditioned oversampling.** Transitions close to the goal are sampled more often than uniform exploration would provide. This intentionally overfits the model near the goal — the region where decisions matter most for return — producing an excellent pretrained model with nearly collapsed epistemic variance there. After a change, FIR's forgetting inflates uncertainty in *all* regions equally; the goal-adjacent overfitting means the model recovers its edge near the goal faster than a uniformly-trained model would.

2. **Variational BNN with a Dirichlet head (discrete) / Gaussian head (continuous).** The trunk is a stack of Bayesian linear layers (softplus activation, Kaiming initialization, KL-to-prior regularization, and a trust-region trick that keeps unvisited (s,a) stuck to the pretrained behavior). The head outputs either Dirichlet concentration α(s,a) over the K transition outcomes (discrete environments) or a Gaussian mean/variance (continuous environments). During pretraining all layers are trained; during the online phase only the head is modified (forget) and optionally retrained — the trunk stays frozen.

## 3.2 Description of the method

**Notation.** After the change notification at time t₀, let p̄(·|s,a) be the BNN's weight-averaged predictive distribution, H(p̄) its (Shannon/differential) entropy, and s_{t+1} the realized next state.

### Step 1 — Surprise (calibrated)

To quantify how wrong the old model is on a single transition, we use a *calibrated* surprise score that is scale-free and has expectation 1 under a correct model:

- Discrete (Dirichlet head):
  $$\delta_t = \frac{-\log \bar p_{s_{t+1}}}{\mathcal H(\bar p)}, \qquad \mathbb E[\delta_t] = 1 \ \text{under a correct model}$$
- Continuous (Gaussian head): normalized Mahalanobis distance
  $$\delta_t = \tfrac{1}{d}(s_{t+1}-\mu)^\top \Sigma^{-1}(s_{t+1}-\mu) \sim \tfrac{1}{d}\chi^2_d$$

A correct model "costs" ~1 per step regardless of environment stochasticity; a genuinely mispredicted transition (regime change) pushes δ well above 1.

### Step 2 — Drift filter (smoothing)

The raw surprise stream is noisy. An equal-weight, *signed* filter with an empirically calibrated baseline accumulates evidence:
$$b = \tfrac{1}{n_b}\sum_{t \in \mathrm{pre}} \delta_t \ (\approx 1), \qquad
\hat\lambda = \tfrac{1}{n}\sum_{i \in \mathrm{post}}(\delta_{n,i} - b), \qquad
\bar\delta = 1 + \hat\lambda.$$
$\bar\delta$ is the smoothed drift estimate: ≈1 when the model still fits, >1 when evidence says "changed". Because it is signed, accumulated evidence *against* a change pulls $\bar\delta$ back below 1 and stops the forgetting — FIR forgets only as long as the change persists.

### Step 3 — Forget (model uncertainty re-inflation)

When drift is detected, the model must unlearn its stale confidence. The retention factor
$$\rho = \frac{1}{1+\max(\hat\lambda, 0)} \in (0,1]$$
decays a scalar retention state r (r ← r·ρ, r₀ = 1), applied to the head's parameters:

- **Discrete (Dirichlet head)** — retract concentrations toward the symmetric prior c (K outcomes), with online counts N on top:
  $$\alpha_k^{\mathrm{eff}} = \underbrace{c + r(\alpha_k^{\mathrm{head}} - c)}_{\text{forget acts here}} + \underbrace{N_k}_{\text{counts}}, \qquad \alpha_0 = \sum_k \alpha_k \xrightarrow{r\to 0} Kc.$$
  The mean is pulled toward uniform and — more importantly — the total concentration α₀ collapses, widening the sampling distribution p ~ Dir(α) (Var[p_k] = p̄_k(1−p̄_k)/(α₀+1)). The re-inflation is realized through the Dirichlet draw; only the head is touched.

- **Continuous (Gaussian head)** — re-inflate the variational weight covariance in parameter space: either bounded retention of the precisions τ ← τ₀ + ρ(τ − τ₀) (self-limiting at the prior variance), or unbounded additive process noise σ² ← σ² + q̂σ₀² (Kalman-style) when the evidence says the parameters moved far. Enlarged σ spreads the sampled networks apart, hence the predictive means.

### Step 4 — Learn / retrain

- **Online (every step, cheap):** conjugate Dirichlet-Multinomial counts N accumulate per (s,a) outcome; with the forgetting retraction, the effective posterior is `retained prior + fresh counts`, so the predictive mean migrates toward the empirical post-change frequencies — the "learn" half of the loop.
- **Batched (once enough data, N ≥ N_threshold):** ELBO updates on the fresh data calibrate the BNN weights. Forget stops firing (δ̄ back to ~1) and retraining takes the lead.

### Planning: confidence-gated CVaR-CEM

At every decision epoch the agent re-plans with a categorical cross-entropy method over action sequences of horizon H:

$$
a^*_{0:H-1} = \arg\max_{a_{0:H-1}} \ \mathrm{CVaR}_\alpha\!\big[Z(a_{0:H-1})\big] + \beta\, U(s_0, a_0),
\qquad Z = \sum_t \gamma^t r_t + \gamma^H V(s_H),
$$

where the imagined returns Z are computed by rolling out each candidate sequence through K posterior transition matrices (Thompson draws — the epistemic axis) × N aleatoric rollouts each. The risk criterion is the empirical CVaR over the worst α-fraction of returns, and U is an optional information bonus on a₀. The same drift signal that drives forgetting also drives the risk level through a confidence score
$$\mathrm{conf} = \underbrace{\min(1, n_{\mathrm{post}}/n_{\mathrm{conf}})}_{\text{data gate}} \cdot \underbrace{e^{-\max(0,\bar\delta-1)/\tau}}_{\text{surprise gate}}, \qquad \alpha = \alpha_{\min} + (\alpha_{\max}-\alpha_{\min})\,\mathrm{conf}.$$
Right after a change, confidence ≈ 0 → α ≈ α_min (CVaR ≈ worst-case return, maximally risk-averse); as fresh evidence accumulates and surprise drops, α rises toward α_max = 1 (risk-neutral mean). The CEM loop refits the per-timestep action distribution to the elite fraction of candidates, warm-started from the model's own greedy policy unrolled under the mean dynamics.

**Algorithm 1** summarizes the complete online procedure.

---

**Algorithm 1: FIR + CVaR-CEM (online, per episode)**
```
Input: pretrained BNN (trunk frozen), drift filter F, planner π_CVaR
1: agent is notified of the change at t₀            ▷ announced change
2: F.reset(); r ← 1; counts N ← 0; confidence ← 0   ▷ drop to max risk-aversion
3: loop for each step t ≥ t₀:
4:     a_t ← π_CVaR.act(s_t)                        ▷ CEM + CVaR(α(confidence)) planning
5:     execute a_t, observe (s_{t+1}, r_t)
6:     δ_t ← surprise(s_t, a_t, s_{t+1})            ▷ Eq. (2) discrete / Eq. (3) continuous
7:     F.update(δ_t)                                ▷ drift filter → δ̄_t = 1 + λ̂
8:     ρ_t ← 1 / max(δ̄_t, 1)                        ▷ retention factor
9:     r ← r · ρ_t                                   ▷ forget: retract head / relax precisions
10:    N ← N + one-hot(s_t, a_t, s_{t+1})            ▷ conjugate online counts (discrete)
11:    confidence ← data-gate · surprise-gate(δ̄_t)
12:    if enough fresh data accumulated:             ▷ batched retraining
13:        ELBO-update(head weights) on the fresh buffer; reset F's post-change window
14: end loop
```

**What the method does, in one sentence.** FIR turns a Bayesian world model into a self-calibrating instrument: it measures how wrong the model is (surprise), smooths and quantifies the error (drift), destroys the stale confidence that would otherwise cause "confident but wrong" plans (forget/inflate), rebuilds the belief from fresh evidence (learn/retrain), and uses the same signal to control how risk-averse the CVaR-CEM planner should be — safe right after a change, efficient once it has adapted.

References