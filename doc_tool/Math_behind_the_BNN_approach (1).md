# Math behind the BNN approach

Yuanhe Li

June 2026

## 1 Math Behind BNN

The environment update is a loop of: surprise → drift → forget → learn. But before this, here is the construct of the BNN:

Bayesian dynamics model is a stack of 3 Bayesian linear layers, with an head depending on the data type (a head is just a layer stacked on top; prediction is categorical: dirichlet head, prediction is continuous: Gaussian head). In the code, dynamics model exposes the interface outside functions need. In each linear layer, tricks (softplus, Kaiming initializations, a KL term between posterior Gaussian and prior Gaussian, a trust-region trick to keep unvisited states stick to the pretrained model) are applied for stabilization. The model currently takes $( s , a )$ as inputs, and spits out $s ^ { \prime }$ predictions for output. The linear layers are not updated throughout run, but only updated during pretraining. What gets updated is only the head layer. Below is roughly how, during pretraining, the layers get updated.

![](images/7b1ec0ced9803d62fc98ff9572794d4b9ae50141de34f1bcc69e55d290526ecd.jpg)  
Figure 1: Lecture Slide from Dr. Matthew Berger

At the start of each run, the agents are notified of env change. When that happens, we first use surprise score to determine the amount of change, then use rectifier to smooth out change based on prior data, finally use forget to update the BNN (or, the head of the BNN). Below are some Claude responses, edited by me:

## 1.1 Calibrated surprise score

To detect regime change we need a per-step statistic that becomes large on genuinely mispredicted transitions but not on the environment’s intrinsic stochasticity. The guiding principle is calibration: under a correctly specified predictive model, the score should have a known, parameter-free scale. Concretely, if the model’s predictive distribution equals the true next-state distribution, so that the realized next state $s _ { t + 1 }$ is drawn from it, we require the surprise score $\delta _ { t }$ follows:

$$
\mathbb {E} \big [ \delta_ {t} \big ] = 1.
$$

General case. Let $\bar { f }$ denote the model’s predictive density over next states at the observed $\left( { { s _ { t } } , { a _ { t } } } \right)$ , and let $s _ { t + 1 }$ <sub>1</sub> be the realized next state. Define the predictive entropy

$$
\mathcal {H} (\bar {f}) = \mathbb {E} _ {s \sim \bar {f}} \bigl [ - \log \bar {f} (s) \bigr ],\tag{1}
$$

i.e. the model’s own expected surprise. Normalizing the realized surprise by this quantity gives the natural candidate

$$
\delta_ {t} = \frac {- \log \bar {f} (s _ {t + 1})}{\mathcal {H} (\bar {f})},\tag{2}
$$

whose numerator is random (it depends on the realized $s _ { t + 1 } )$ while its denominator is a constant given $\left( { { s _ { t } } , { a _ { t } } } \right)$ ; taking the expectation of the numerator via (1) yields $\mathbb { E } [ \delta _ { t } ] = 1$

In continuous spaces, however, this normalization is generally unsuitable: the diferential entropy $\mathcal { H } ( \bar { f } ) = h ( \bar { f } )$ is not invariant under reparameterization of the state space (unlike the Shannon entropy of a discrete distribution) and may be zero or negative, in which case the ratio in (2) is undefined or signinverted. For a Gaussian predictive distribution $\textstyle { \mathcal { N } } ( \mu , \Sigma )$ in d dimensions we instead use the normalized Mahalanobis distance

$$
\delta_ {t} = \frac {1}{d} (s _ {t + 1} - \mu) ^ {\top} \Sigma^ {- 1} (s _ {t + 1} - \mu), \quad \delta_ {t} \sim \frac {1}{d} \chi_ {d} ^ {2}\tag{3}
$$

under the correctly specified model, which is afine-invariant and satisfies $\mathbb { E } [ \delta _ { t } ] =$ 1 (Note that equation 3 is an expansion of equation 2 under the Gaussian assumption).

Discrete case. When the state space is discrete, as in the FrozenLake environment, the predictive distribution is a categorical p, with $\bar { p } _ { s ^ { \prime } }$ the predicted probability of next state $s ^ { \prime } .$ , and the pathologies above vanish. The predictive entropy specializes to the Shannon entropy

$$
\mathcal {H} (\overline {{\mathbf {p}}}) = - \sum_ {s ^ {\prime}} \bar {p} _ {s ^ {\prime}} \log \bar {p} _ {s ^ {\prime}},
$$

which is nonnegative and invariant to relabeling of states, so the entropy normalization is well defined (except in the degenerate deterministic case $\mathcal { H } = 0$ 2 which we exclude). The calibrated surprise (2) then reads

$$
\delta_ {t} = \frac {- \log \bar {p} _ {s _ {t + 1}}}{\mathcal {H} (\overline {{\mathbf {p}}})},\tag{4}
$$

where p is the weight-averaged predictive categorical distribution at $( s _ { t } , a _ { t } )$ and $s _ { t + 1 }$ is the realized next state. By the discrete form of $( 1 ) , \mathbb { E } \left[ - \log \bar { p } _ { s _ { t + 1 } } \right] =$ $\mathcal { H } ( \overline { { \mathbf { p } } } )$ , so the statistic again satisfies $\mathbb { E } [ \delta _ { t } ] = 1$ under the correctly specified model. Equation (4) is the form used in our discrete-state experiments.

## 1.2 Drift filtering

The surprise stream $\delta _ { t }$ is noisy; we filter it into a smoothed drift estimate $\bar { \delta }$ with baseline 1. We use an equal-weight, signed estimator with an empirically calibrated baseline. What this means is we calculated the expected value of the surprise score (averaging over a window!):

$$
b = \frac {1}{n _ {b}} \sum_ {t \in \mathrm{pre}} \delta_ {t} \big (\approx 1 \big),\tag{5}
$$

and after the change notification, we average them out:

$$
\hat {\lambda} = \frac {1}{n} \sum_ {i \in \text {post}} \big (\delta_ {n, i} - b \big), \quad \bar {\delta} = 1 + \hat {\lambda}.\tag{6}
$$

The estimator is signed so that accumulated evidence against a change $( \delta _ { n } < b )$ pulls <sup>¯</sup>δ back below 1.

## 1.3 Forget: Model Uncertainty Re-inflation

When drift is detected the model must unlearn its stale confidence, otherwise it would produce confidence but wrong predictions. We first define $\rho$ as:

$$
\rho = \frac {1}{1 + \max (\hat {\lambda} , 0)} \in (0, 1 ]\tag{7}
$$

## 1.3.1 Discrete model: retract the Dirichlet concentrations

The Dirichlet-head model re-inflates in outcome space. A scalar retention state $r _ { 0 } = 1$ , is decayed multiplicatively, $r _ { t + 1 } \gets r _ { t } \rho _ { : }$ , and applied as an afine retract of the head’s concentrations toward a symmetric prior c1 (K outcome categories):

$$
\alpha_ {k} ^ {\mathrm{eff}} = \underbrace {c + r (\alpha_ {k} ^ {\mathrm{head}} - c)} _ {\text {forget acts here}} + \underbrace {N _ {k}} _ {\text {counts}}. \quad \alpha_ {0} = \sum_ {k} \alpha_ {k} \xrightarrow {r \to 0} K c.\tag{8}
$$

Under (8) the mean is pulled toward uniform and, more importantly, α<sub>0</sub> collapses (because r is too small), which widens the sampling distribution of the transition probabilities:

$$
\boldsymbol {p} \sim \mathrm{Dir} (\boldsymbol {\alpha}), \qquad \mathrm{Var} [ p _ {k} ] = \frac {\bar {p} _ {k} (1 - \bar {p} _ {k})}{\alpha_ {0} + 1}.\tag{9}
$$

The re-inflation is thus realized through the Dirichlet draw: we resets the draw by controlling r . Note that only the head layer of the BNN is modified.

## 1.3.2 Continuous model: re-inflate the weight covariance

The Gaussian-head model re-inflates in parameter space, acting directly on the variational posterior $q ( w ) = \mathcal { N } ( \mu , \sigma ^ { 2 } )$ of every weight (prior $\mathcal { N } ( m , \sigma _ { 0 } ^ { 2 } ) ,$ , in one of two modes.

(a) Retention (bounded, default). The pull toward the prior is applied to the precisions $\tau = 1 / \sigma ^ { 2 } , \tau _ { 0 } = 1 / \sigma _ { 0 } ^ { 2 }$

$$
\tau \leftarrow \tau_ {0} + \rho (\tau - \tau_ {0}),\tag{10}
$$

so $\sigma ^ { 2 }$ relaxes toward—but never past—the prior variance $\sigma _ { 0 } ^ { 2 } ;$ repeated application is self-limiting.

(b) Additive process noise (unbounded). With ˆq proportional to the filtered squared-drift estimate λ<sup>ˆ</sup> (capped per application),

$$
\sigma^ {2} \leftarrow \sigma^ {2} + \hat {q} \sigma_ {0} ^ {2},\tag{11}
$$

the Kalman-style covariance update $P  P + Q ;$ : a large measured drift can push the epistemic uncertainty past the prior, appropriate when the evidence says the parameters moved far.

Here the re-inflation is realized through the weight draw $w = \mu { + } \sigma \varepsilon { : }$ : enlarged σ spreads the sampled networks, and hence the predictive means, apart.

## 2 Nice Things

The guarantee is purely of the mechanics of surprise $ \mathrm { r e c t i f y }  ( \mathrm { r e } - ) \mathrm { i n f l a t e }$ Not update.

Theorem 1. (Without retraining) After an environment change from $p _ { 0 }$ to $p _ { 1 }$ , the number of new samples n for the model to be ε-accurate follows: $\begin{array} { r } { n ( \varepsilon ) \leq \frac { \sqrt { 2 } m | | p _ { 0 } - p _ { 1 } | | } { \varepsilon } + \frac { 2 ( 1 - | | p _ { 1 } | | ) ^ { 2 } } { \varepsilon ^ { 2 } } } \end{array}$ in the discrete case.

Proof. Preliminary: to update category i of a Dirichlet distribution, one add the new count to the belief like $\alpha _ { i } = \alpha _ { i } + c _ { j }$ . Its posterior mean updates as

$$
\mathbb {E} [ \hat {p} _ {i} ] = \frac {\alpha_ {i} + c _ {i}}{\sum_ {j = 1} ^ {K} (\alpha_ {j} + c _ {j})}
$$

Now, per equation (8), we first define $\begin{array} { r } { S = \sum _ { k = 1 } ^ { K } [ c + r ( \alpha _ { k } - c ) + N _ { k } ] } \end{array}$ . If define $\begin{array} { r } { A = \sum _ { k } \alpha _ { k } , n = \sum _ { k } N _ { k } } \end{array}$ , we have $S = K c + r ( A - K c ) + n$ . For simplification, define $m = r ( A - K c )$ . Then, we have

$$
\mathbb {E} [ \hat {p} ] = \frac {1}{K c + m + n} \cdot ([ c, \dots , c ] ^ {T} + r ([ \alpha_ {1}, \dots , \alpha_ {K} ] ^ {T} - c) + [ N _ {k}, \dots , N _ {k} ] ^ {T})
$$

Claude redefines $\begin{array} { r } { \mathbb { E } [ \hat { p } ] = \frac { K c u + m p _ { 0 } + n \hat { p } _ { e m p } } { K c + m + n } \approx \frac { m p _ { 0 } + n \hat { p } _ { e m p } } { m + n } } \end{array}$ , where $\begin{array} { r } { p _ { 0 , k } = \frac { r ( \alpha _ { k } - c ) } { m } } \end{array}$ is the old (staled) data and $\begin{array} { r } { \hat { p } _ { e m p , k } = \frac { N _ { k } } { n } } \end{array}$ (the $^ { 6 } , \mathrm { k } ^ { \prime \prime }$ denotes the k-th entry of the vector).

Then, we want to bound the model’s expected error by our approximated form,

$$
\begin{array}{r l} \hat {p} - p _ {1} & \approx \frac {m p _ {0} + n \hat {p} _ {\mathrm{emp}} - (m + n) p _ {1}}{m + n} \\ & = \left(\frac {m}{m + n}\right) (p _ {0} - p _ {1}) + \left(\frac {n}{m + n}\right) (\hat {p} _ {\mathrm{emp}} - p _ {1}) \end{array}
$$

Let $w _ { 0 } = m / ( m + n ) , w _ { 1 } = n / ( m + n )$ , we take the L2 norm:

$$
\| \hat {p} - p _ {1} \| ^ {2} = w _ {0} ^ {2} \| p _ {0} - p _ {1} \| ^ {2} + w _ {1} ^ {2} \| \hat {p} _ {\mathrm{emp}} - p _ {1} \| ^ {2} + 2 w _ {0} w _ {1} (p _ {0} - p _ {1}) \cdot (\hat {p} _ {\mathrm{emp}} - p _ {1})
$$

$$
\mathbb {E} \left\| \hat {p} - p _ {1} \right\| ^ {2} = w _ {0} ^ {2} \| p _ {0} - p _ {1} \| ^ {2} + w _ {1} ^ {2} \mathbb {E} \left\| \hat {p} _ {\mathrm{emp}} - p _ {1} \right\| ^ {2} + 2 w _ {0} w _ {1} (p _ {0} - p _ {1}) \cdot \mathbb {E} [ \hat {p} _ {\mathrm{emp}} - p _ {1} ]
$$

Now, as we have more data, $\hat { p } _ { e m p }  p _ { 1 }$ , making the last term 0:

$$
\mathbb {E} \left\| \hat {p} - p _ {1} \right\| ^ {2} = w _ {0} ^ {2} \| p _ {0} - p _ {1} \| ^ {2} + w _ {1} ^ {2} \mathbb {E} \left\| \hat {p} _ {\mathrm{emp}} - p _ {1} \right\| ^ {2}
$$

Now, what $\mathrm { i s } \ \mathbb { E } \| \hat { p } _ { \mathrm { e m p } } - p _ { 1 } \| ^ { 2 } ?$ Well, the total expected squared norm of the error vector is simply the sum of the variances across all K categories:

$$
\mathbb {E} \left\| \hat {p} _ {\mathrm{emp}} - p _ {1} \right\| ^ {2} = \sum_ {k = 1} ^ {K} \frac {p _ {1 , k} (1 - p _ {1 , k})}{n} = \frac {1}{n} \left(\sum_ {k = 1} ^ {K} p _ {1, k} - \sum_ {k = 1} ^ {K} p _ {1, k} ^ {2}\right)
$$

The first term sums to 1, and second term is just $| | p _ { 1 } | | ^ { 2 }$ . So, we have:

$$
\mathbb {E} \left\| \hat {p} - p _ {1} \right\| ^ {2} = \left(\frac {m}{m + n}\right) ^ {2} \left\| p _ {0} - p _ {1} \right\| ^ {2} + \frac {n}{(m + n) ^ {2}} \left(1 - \left\| p _ {1} \right\| ^ {2}\right)
$$

This is the BNN’s error. Now, we want to see the minimum number of new samples $( n )$ required to guarantee that the expected squared error falls below a target threshold, $\varepsilon ^ { 2 }$ . rewrite:

$$
\mathbb {E} \left\| \hat {p} - p _ {1} \right\| ^ {2} = \left(\frac {m}{m + n}\right) ^ {2} \Delta^ {2} + \frac {n}{(m + n) ^ {2}} \sigma_ {p} ^ {2} \leq \varepsilon^ {2}.
$$

Letting each term $\begin{array} { r } { \le \frac { \varepsilon ^ { 2 } } { 2 } \mathrm { ~ g i v e s ~ } n ( \varepsilon ) \le \frac { \sqrt 2 m \Delta } { \varepsilon } + \frac { 2 \sigma _ { p } ^ { 2 } } { \varepsilon ^ { 2 } } } \end{array}$

First, we can see the change doesn’t have to be bounded. $\Delta$ can be arbitrarily large in our approach. Second, we see recovery time (number of new samples) is linear in retained stale mass. Second, what does this bound mean, in relation to our model? This requires us to expand $m$ . We take a look at an singular update:

Claim 1. In the discrete case, we forget more exactly where old evidence is more wrong. In math, the higher $D _ { K L } ( p _ { 1 } | | p _ { 0 } )$ is, the rate of decay of the retain factor, $\textstyle { \frac { d r _ { t } } { d t } }$ , should be higher (See equation 8, $r _ { t + 1 } \gets r _ { t } \rho , r _ { 0 } = 1 )$

Proof. In one iteration, $m = \rho ( A - K c )$ and we could get:

$$
m = (A - K c) \min \bigl (1, \frac {1}{\delta} \bigr) = (A - K c) \min \Biggl (1, \frac {H (p _ {0})}{- \mathbb {E} _ {p _ {1}} [ \log p _ {0} (s ^ {\prime}) ]} \Biggr)
$$

and $- \mathbb { E } _ { p _ { 1 } } [ \log p _ { 0 } ( s ^ { \prime } ) ] = C E ( p _ { 1 } , p _ { 0 } ) = H ( p _ { 1 } ) + D _ { K L } ( p _ { 1 } | | p _ { 0 } ) .$

Now, the denominator is just the cross entropy of $p _ { 1 }$ and $p _ { 0 }$ . Why?

$$
\mathrm{CE} (p _ {1}, p _ {0}) = - \sum_ {k} p _ {1, k} \log p _ {0, k} = - \sum_ {k} p _ {1, k} \log p _ {1, k} + \sum_ {k} p _ {1, k} \big (\log p _ {1, k} - \log p _ {0, k} \big).
$$

The first group i $\begin{array} { r } { \mathrm { ~ s ~ } - \sum _ { k } p _ { 1 , k } \log p _ { 1 , k } = H ( p _ { 1 } ) } \end{array}$ , the entropy of the new regime. The second group is $\sum _ { k } p _ { 1 , k }$ log $\begin{array} { r } { \frac { p _ { 1 , k } } { p _ { 0 , k } } = D _ { \mathrm { K L } } ( p _ { 1 } \| p _ { 0 } ) } \end{array}$ . So

$$
\mathrm{CE} (p _ {1}, p _ {0}) = H (p _ {1}) + D _ {\mathrm{KL}} (p _ {1} \| p _ {0}).
$$

Now, r decays by a factor of $\rho ,$ and so roughly $\begin{array} { r } { r _ { t } = ( \frac { H ( p _ { 0 } ) } { C E } ) ^ { t } } \end{array}$ , and that

$$
\frac {d r _ {t}}{d t} = r _ {t} \log \frac {H (p _ {0})}{D _ {\mathrm{KL}} (p _ {1} \| p _ {0}) + H (p _ {1})}
$$

## 3 Results

## 3.1 FrozenLake

This is without retrain (only

Table 1: Expected return over 100 trials (undiscounted, $\gamma _ { \mathrm { e v a l } } = 1 )$ . Columns: planner CVaR risk level α. Static rows: slip level θ (intended-move prob; the slip distribution is $[ \theta , { \frac { 1 - \theta } { 2 } } , { \frac { 1 - \theta } { 2 } } ] )$ $\mathrm { ^ { 6 4 } h o l e ^ { 3 7 } / ^ { 4 6 } g o a l ^ { 3 7 } }$ give the terminal-reward scoring of the reported return ; the change schedule lists (step, intended-prob). No truncation.

<table><tr><td rowspan="2">scenario</td><td colspan="10">CVaR risk level α</td></tr><tr><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>0.8</td><td>0.9</td><td>1</td></tr><tr><td colspan="11">hole = 0, goal = 1 (static)</td></tr><tr><td>θ = 0.9</td><td>0.71</td><td>0.70</td><td>0.68</td><td>0.69</td><td>0.69</td><td>0.69</td><td>0.69</td><td>0.70</td><td>0.72</td><td>0.71</td></tr><tr><td>θ = 0.7</td><td>0.55</td><td>0.55</td><td>0.50</td><td>0.47</td><td>0.43</td><td>0.49</td><td>0.46</td><td>0.47</td><td>0.42</td><td>0.52</td></tr><tr><td>θ = 0.5</td><td>0.62</td><td>0.52</td><td>0.48</td><td>0.31</td><td>0.34</td><td>0.32</td><td>0.29</td><td>0.32</td><td>0.32</td><td>0.34</td></tr><tr><td>θ = 0.3</td><td>0.49</td><td>0.49</td><td>0.36</td><td>0.23</td><td>0.27</td><td>0.29</td><td>0.24</td><td>0.21</td><td>0.21</td><td>0.25</td></tr><tr><td>θ = 0.1</td><td>0.62</td><td>0.59</td><td>0.38</td><td>0.46</td><td>0.34</td><td>0.35</td><td>0.28</td><td>0.27</td><td>0.33</td><td>0.32</td></tr><tr><td colspan="11">hole = -1, goal = 1 (static)</td></tr><tr><td>θ = 0.9</td><td>0.55</td><td>0.50</td><td>0.48</td><td>0.51</td><td>0.50</td><td>0.49</td><td>0.51</td><td>0.51</td><td>0.54</td><td>0.54</td></tr><tr><td>θ = 0.7</td><td>0.14</td><td>0.13</td><td>0.00</td><td>-0.05</td><td>-0.13</td><td>-0.01</td><td>-0.06</td><td>-0.05</td><td>-0.14</td><td>0.05</td></tr><tr><td>θ = 0.5</td><td>0.24</td><td>0.04</td><td>-0.04</td><td>-0.27</td><td>-0.32</td><td>-0.36</td><td>-0.41</td><td>-0.35</td><td>-0.36</td><td>-0.31</td></tr><tr><td>θ = 0.3</td><td>-0.02</td><td>-0.02</td><td>-0.28</td><td>-0.54</td><td>-0.46</td><td>-0.42</td><td>-0.52</td><td>-0.58</td><td>-0.58</td><td>-0.50</td></tr><tr><td>θ = 0.1</td><td>0.24</td><td>0.18</td><td>-0.23</td><td>-0.08</td><td>-0.32</td><td>-0.29</td><td>-0.44</td><td>-0.45</td><td>-0.33</td><td>-0.35</td></tr><tr><td colspan="11">hole = -10, goal = 1 (static)</td></tr><tr><td>θ = 0.9</td><td>-0.89</td><td>-1.30</td><td>-1.32</td><td>-1.11</td><td>-1.21</td><td>-1.31</td><td>-1.11</td><td>-1.20</td><td>-1.08</td><td>-0.99</td></tr><tr><td>θ = 0.7</td><td>-3.55</td><td>-3.65</td><td>-4.50</td><td>-4.73</td><td>-5.17</td><td>-4.51</td><td>-4.74</td><td>-4.73</td><td>-5.18</td><td>-4.18</td></tr><tr><td>θ = 0.5</td><td>-3.18</td><td>-4.28</td><td>-4.72</td><td>-5.49</td><td>-6.26</td><td>-6.48</td><td>-6.71</td><td>-6.38</td><td>-6.48</td><td>-6.16</td></tr><tr><td>θ = 0.3</td><td>-4.61</td><td>-4.61</td><td>-6.04</td><td>-7.47</td><td>-7.03</td><td>-6.81</td><td>-7.36</td><td>-7.69</td><td>-7.69</td><td>-7.25</td></tr><tr><td>θ = 0.1</td><td>-3.18</td><td>-3.51</td><td>-5.72</td><td>-4.94</td><td>-6.26</td><td>-6.05</td><td>-6.92</td><td>-6.93</td><td>-6.27</td><td>-6.38</td></tr><tr><td colspan="11">change schedule [(0, 0.8), (3, 0.7)]</td></tr><tr><td>hole = 0, goal = 1</td><td>0.74</td><td>0.68</td><td>0.63</td><td>0.56</td><td>0.53</td><td>0.57</td><td>0.55</td><td>0.56</td><td>0.57</td><td>0.56</td></tr><tr><td>hole = -1, goal = 1</td><td>0.48</td><td>0.37</td><td>0.27</td><td>0.13</td><td>0.07</td><td>0.15</td><td>0.11</td><td>0.14</td><td>0.16</td><td>0.13</td></tr><tr><td>hole = -10, goal = 1</td><td>-1.86</td><td>-2.42</td><td>-2.97</td><td>-3.74</td><td>-4.07</td><td>-3.63</td><td>-3.85</td><td>-3.64</td><td>-3.53</td><td>-3.74</td></tr></table>

Table 2: Pendulum: mean undiscounted return over the trials (200-step episodes, no discounting; reward is a negative cost so higher is better). The BNN is pretrained on the default env (mass 1, g 10, goal-weighted) and the mass/gravity shifts to the row value mid-episode; columns are the planner CVaR risk level α. The best α per row is in bold.

<table><tr><td rowspan="2">env shift</td><td colspan="10">CVaR risk level  $\alpha$ </td></tr><tr><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>0.8</td><td>0.9</td><td>1</td></tr><tr><td colspan="11">mass shift 1→m (g=10)</td></tr><tr><td>m = 1</td><td>-466.0</td><td>-466.0</td><td>-453.7</td><td>-466.0</td><td>-465.5</td><td>-464.5</td><td>-464.5</td><td>-466.0</td><td>-469.3</td><td>-468.5</td></tr><tr><td>m = 1.5</td><td>-820.8</td><td>-820.8</td><td>-827.6</td><td>-820.8</td><td>-824.3</td><td>-823.3</td><td>-823.2</td><td>-820.8</td><td>-825.9</td><td>-827.4</td></tr><tr><td>m = 2</td><td>-843.6</td><td>-</td><td>-846.8</td><td>-843.6</td><td>-</td><td>-845.6</td><td>-845.5</td><td>-843.6</td><td>-848.9</td><td>-850.0</td></tr><tr><td>m = 3</td><td>-853.0</td><td>-853.0</td><td>-854.6</td><td>-853.0</td><td>-854.3</td><td>-853.8</td><td>-853.8</td><td>-853.0</td><td>-855.6</td><td>-855.8</td></tr><tr><td>m = 5</td><td>-</td><td>-856.7</td><td>-857.3</td><td>-856.7</td><td>-858.3</td><td>-857.9</td><td>-</td><td>-856.7</td><td>-858.9</td><td>-859.4</td></tr><tr><td colspan="11">gravity shift 10→g (m=1)</td></tr><tr><td>g = 12</td><td>-652.6</td><td>-</td><td>-680.1</td><td>-652.6</td><td>-669.3</td><td>-680.4</td><td>-680.6</td><td>-652.6</td><td>-688.0</td><td>-677.3</td></tr><tr><td>g = 15</td><td>-912.1</td><td>-912.1</td><td>-912.1</td><td>-912.1</td><td>-913.4</td><td>-913.1</td><td>-913.0</td><td>-912.1</td><td>-912.4</td><td>-913.6</td></tr><tr><td>g = 20</td><td>-1030.1</td><td>-1030.1</td><td>-</td><td>-1030.1</td><td>-1030.5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-1029.7</td></tr></table>