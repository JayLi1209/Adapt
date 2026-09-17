# Non-Stationary Markov Decision Processes a Worst-Case Approach using Model-Based Reinforcement Learning

Erwan Lecarpentier Université de Toulouse ONERA - The French Aerospace Lab erwan.lecarpentier@isae-supaero.fr

Emmanuel Rachelson Université de Toulouse ISAE-SUPAERO emmanuel.rachelson@isae-supaero.fr

## Abstract

This work tackles the problem of robust planning in non-stationary stochastic environments. We study Markov Decision Processes (MDPs) evolving over time and consider Model-Based Reinforcement Learning algorithms in this setting. We make two hypotheses: 1) the environment evolves continuously with a bounded evolution rate; 2) a current model is known at each decision epoch but not its evolution. Our contribution can be presented in four points. 1) we define a specific class of MDPs that we call Non-Stationary MDPs (NSMDPs). We introduce the notion of regular evolution by making an hypothesis of Lipschitz-Continuity on the transition and reward functions w.r.t. time; 2) we consider a planning agent using the current model of the environment but unaware of its future evolution. This leads us to consider a worst-case method where the environment is seen as an adversarial agent; 3) following this approach, we propose the Risk-Averse Tree-Search (RATS) algorithm, a Model-Based method similar to minimax search; 4) we illustrate the benefits brought by RATS empirically and compare its performance with reference Model-Based algorithms.

## 1 Introduction

One of the hot topics of modern Artificial Intelligence (AI) is the ability for an agent to adapt its behavior to changing tasks. In the literature, this problem is often linked to the setting of Lifelong Reinforcement Learning (LRL) [Silver et al., 2013, Abel et al., 2018a,b] and learning in non-stationary environments [Choi et al., 1999, Jaulmes et al., 2005, Hadoux, 2015]. In LRL, the tasks presented to the agent change sequentially at discrete transition epochs [Silver et al., 2013]. Similarly, the nonstationary environments considered in the literature often evolve abruptly [Hadoux, 2015, Hadoux et al., 2014, Doya et al., 2002, Da Silva et al., 2006, Choi et al., 1999, 2000, 2001, Campo et al., 1991, Wiering, 2001]. In this paper, we investigate environments continuously changing over time that we call Non-Stationary Markov Decision Processes (NSMDPs). In this setting, it is realistic to bound the evolution rate of the environment using a Lipschitz Continuity (LC) assumption.

Model-based Reinforcement Learning approaches [Sutton et al., 1998] benefit from the knowledge of a model allowing them to reach impressive performances, as demonstrated by the Monte Carlo Tree Search (MCTS) algorithm [Silver et al., 2016]. In this matter, the necessity to have access to a model is a great concern of AI [Asadi et al., 2018, Jaulmes et al., 2005, Doya et al., 2002, Da Silva et al., 2006]. In the context of NSMDPs, we assume that an agent is provided with a snapshot model when its action is computed. By this, we mean that it only has access to the current model of the environment but not its future evolution, as if it took a photograph but would be unable to predict how it is going to evolve. This hypothesis is realistic, because many environments have a tractable state while their future evolution is hard to predict [Da Silva et al., 2006, Wiering, 2001]. In order to solve

LC-NSMDPs, we propose a method that considers the worst-case possible evolution of the model and performs planning w.r.t. this model. This is equivalent to considering Nature as an adversarial agent. The paper is organized as follows: first we describe the NSMDP setting and the regularity assumption (Section $^ { 2 ) }$ ; then we outline related works (Section 3); follows the explanation of the worst-case approach proposed in this paper (Section 4); then we describe an algorithm reflecting this approach (Section 5); finally we illustrate its behavior empirically (Section 6).

## 2 Non-Stationary Markov Decision Processes

To define a Non-Stationary Markov Decision Process (NSMDP), we revert to the initial MDP model introduced by Puterman [2014], where the transition and reward functions depend on time.

Definition 1. NSMDP. An NSMDP is an MDP whose transition and reward functions depend on the decision epoch. It is defined by a 5-tuple $\{ S , T , \mathcal { A } , ( p _ { t } ) _ { t \in \mathcal { T } } , ( r _ { t } ) _ { t \in \mathcal { T } } \}$ where  is a state space; $\mathcal { T } \equiv \{ 1 , 2 , \dots , N \}$ is the set of decision epochs with $N \leq + \infty ; A$ is an action space; $p _ { t } ( s ^ { \prime } \mid s , a )$ is the probability of reaching state $s ^ { \prime }$ while performing action a at decision epoch t in state $s ; r _ { t } ( s , a , s ^ { \prime } )$ is the scalar reward associated to the transition from s to $s ^ { \prime }$ with action a at decision epoch t.

This definition can be viewed as that of a stationary MDP whose state space has been enhanced with time. While this addition is trivial in episodic tasks where an agent is given the opportunity to interact several times with the same MDP, it is different when the experience is unique. Indeed, no exploration is allowed along the temporal axis. Within a stationary, infinite-horizon MDP with a discounted criterion, it is proven that there exists a Markovian deterministic stationary policy [Puterman, 2014]. It is not the case within NSMDPs where the optimal policy is non-stationary in the most general case. Additionally, we define the expected reward received when taking action a at state s and decision epoch t as $\smash { \dot { R } _ { t } ( s , a ) = \mathbb { E } _ { s ^ { \prime } \sim p _ { t } ( \cdot \vert s , a ) } \left[ r _ { t } ( s , a , s ^ { \prime } ) \right] }$ . Without loss of generality, we assume the reward function to be bounded between 1 and 1. In this paper, we consider discrete time decision processes with constant transition durations, which imply deterministic decision times in Definition 1. This assumption is mild since many discrete time sequential decision problems follow that assumption. A non-stationary policy π is a sequence of decision rules $\pi _ { t }$ which map states to actions (or distributions over actions). For a stochastic non-stationary policy $\pi _ { t } ( a \mid s )$ , the value of a state s at decision epoch t within an infinite horizon NSMDP is defined, with $\gamma \in [ 0 , 1 )$ a discount factor, by:

$$
V_{t}^{\pi}(s) = \mathbb{E}\left[\sum_{i = t}^{\infty}\gamma^{i - t}R_{i}(s_{i},a_{i})\Big|  s_{t} = s, a_{i}\sim \pi_{i}(\cdot \mid s_{i}), s_{i + 1}\sim p_{i}(\cdot \mid s_{i},a_{i})\right],
$$

The definition of the state-action value function $Q _ { t } ^ { \pi }$ for π at decision epoch t is straightforward:

$$
Q _ {t} ^ {\pi} (s, a) = R _ {t} (s, a) + \gamma \underset {s ^ {\prime} \sim p _ {t} (\cdot | s, a)} {\mathbb {E}} \left[ V _ {t + 1} ^ {\pi} (s ^ {\prime}) \right].
$$

Overall, we defined an NSMDP as an MDP where we stress out the distinction between state, time, and decision epoch due to the inability for an agent to explore the temporal axis at will. This distinction is particularly relevant for non-episodic tasks, i.e. when there is no possibility to re-experience the same MDP starting from a prior date.

The regularity hypothesis. Many real-world problems can be modeled as an NSMDP. For instance, the problem of path planning for a glider immersed in a non-stationary atmosphere [Chung et al., 2015, Lecarpentier et al., 2017], or that of vehicle routing in dynamic traffic congestion. Realistically, we consider that the expected reward and transition functions do not evolve arbitrarily fast over time. Conversely, if such an assumption was not made, a chaotic evolution of the NSMDP would be allowed which is both unrealistic and hard to solve. Hence, we assume that changes occur slowly over time. Mathematically, we formalize this hypothesis by bounding the evolution rate of the transition and expected reward functions, using the notion of Lipschitz Continuity (LC).

Definition 2. Lipschitz Continuity. Let $( X , d _ { X } )$ and $( Y , d _ { Y } )$ be two metric spaces and $f : X \to Y ,$ f is L-Lipschitz Continuous (L-LC) with L R<sup>+</sup> $i f f { d _ { Y } } ( f ( x ) , f ( \hat { x } ) ) \leq L { \bar { d _ { X } } } ( x , \hat { x } ) , \forall ( x , \hat { x } ) \in X ^ { 2 } .$ L is called a Lipschitz constant of the function $f .$

We apply this hypothesis to the transition and reward functions of an NSMDP so that those functions are LC w.r.t. time. For the transition function, this leads to the consideration of a metric between probability density functions. For that purpose, we use the 1-Wasserstein distance [Villani, 2008].

The minimal distance to from one distribution to another with same mass

Definition 3. 1-Wasserstein distance. Let $( X , d _ { X } )$ be a Polish metric space, $\mu , \nu$ any probability measures on X, $\Pi ( \mu , \nu )$ the set of joint distributions on $X \times X$ with marginals $\mu$ and ν. The 1-Wasserstein distance between $\mu$ and ν is $\begin{array} { r } { W _ { 1 } ( \mu , \nu ) = \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int _ { X \times X } d _ { X } \overline { { ( x , y ) } } d \pi ( x , y ) } \end{array}$

The choice of the Wasserstein distance is motivated by the fact that it quantifies the distance between two distributions in a physical manner, respectful of the topology of the measured space [Dabney et al., 2018, Asadi et al., 2018]. First, it is sensitive to the difference between the supports of the distributions. Comparatively, the Kullback-Leibler divergence between distributions with disjoint supports is infinite. Secondly, if one consider two regions of the support where two distributions differ, the Wasserstein distance is sensitive to the distance between the elements of those regions. Comparatively, the total-variation metric is the same regardless of this distance.

Definition 4. $( L _ { p } , L _ { r } ) { \bf - } L C { \bf - } N S M D P .$ . An $( L _ { p } , L _ { r } ) – L C – N S M D P$ is an NSMDP whose transition and reward functions are respectively $L _ { p ^ { - L C } }$ and $L _ { r ^ { - } L C }$ w.r.t. time, i. $\begin{array} { r } { \ ? _ { \cdot } , \forall ( t , { \hat { t } } , s , s ^ { \prime } , a ) \in { \mathcal { T } } ^ { 2 } \times { \mathcal { S } } ^ { 2 } \times { \mathcal { A } } , } \end{array}$

$$
W _ {1} (p _ {t} (\cdot \mid s, a), p _ {\hat {t}} (\cdot \mid s, a)) \leq L _ {p} | t - \hat {t} | \quad a n d \quad | r _ {t} (s, a, s ^ {\prime}) - r _ {\hat {t}} (s, a, s ^ {\prime}) | \leq L _ {r} | t - \hat {t} |.
$$

One should remark that the LC property should be defined with respect to actual decision times and not decision epoch indexes for the sake of realism. In the present case, both have the same value, and we choose to keep this convention for clarity. Our results however extend easily to the case where indexes and times do not coincide. From now on, we consider $( L _ { p } , L _ { r } ) \mathrm { - L C - N S M D P s }$ , making Lipschitz Continuity our regularity property. Notice that R is defined as a convex combination of r by the probability measure $p .$ As a result, the notion of Lipschitz Continuity of R is strongly related to that of r and $p$ as showed by Property 1. All the proofs of the paper can be found in the Appendix.

Property 1. Given an $( L _ { p } , L _ { r } ) – L C – N S M D P ,$ , the expected reward function $R _ { t } : \ s , a \mapsto $ $\mathbb { E } _ { s ^ { \prime } \sim p _ { t } ( \cdot | s , a ) } \left\{ r _ { t } ( s , a , s ^ { \prime } ) \right\}$ is $\mathrm { \bar { \it L } } _ { R ^ { - } } L C$ with $L _ { R } = L _ { r } + L _ { p } . $

This result shows $R \ ' \mathrm { s }$ evolution rate is conditioned by the evolution rates of r and $p .$ It allows to work either with the reward function r or its expectation R, benefiting from the same LC property.

## 3 Related work

Iyengar [2005] introduced the framework of robust MDPs, where the transition function is allowed to evolve within a set of functions due to uncertainty. This differs from our work in two fundamental aspects: 1) we consider uncertainty in the reward model as well; 2) we use a stronger Lipschitz formulation on the set of possible transition and reward functions, this last point being motivated by its relevance to the non-stationary setting. Szita et al. [2002] also consider the robust MDP setting and adopt a different constraint hypothesis on the set of possible functions than our LC assumption. They control the total variation distance of transition functions from subsequent decision epochs by a scalar value. Those slowly changing environments allow model-free RL algorithms such as Q-Learning to find near optimal policies. Lim et al. [2013] consider learning in robust MDPs where the model evolves in an adversarial manner for a subset of ${ \mathcal { S } } \times { \mathcal { A } } .$ In that setting, they propose to learn to what extent the adversary can modify the model and to deduce a behavior close to the minimax policy. Even-Dar et al. [2009] studied the case of non-stationary reward functions with fixed transition models. No assumption is made on the set of possible functions and they propose an algorithm achieving sub-linear regret w.r.t. the best stationary policy. Dick et al. [2014] viewed a similar setting from the perspective of online linear optimization. Csáji and Monostori [2008] studied the NSMDP setting with an assumption of reward and transition functions varying in a neighborhood of a reference reward-transition function pair. Finally, Abbasi et al. [2013] address the adversarial NSMDP setting with a mixing assumption constraint instead of the LC assumption we make.

Non-stationary environments also have been studied through the framework of Hidden Mode MDPs (HM-MDP) introduced by Choi et al. [1999]. This is a special class of Partially Observable MDPs (POMDPs) [Kaelbling et al., 1998] where a hidden mode indexes a latent stationary MDP within which the agent evolves. Similarly to the context of LRL, the agent experiences a series of different MDPs over time. In this setting, Choi et al. [1999, 2000] proposed methods to learn the different models of the latent stationary MDPs. Doya et al. [2002] built a modular architecture switching between models and policies when a change is detected. Similarly, Wiering [2001], Da Silva et al. [2006], Hadoux et al. [2014] proposed a method tracking the switching occurrence and re-planning if needed. Overall, as in LRL, the HM-MDP setting considers abrupt evolution of the transition and reward functions whereas we consider a continuous one. Other settings have been considered, as by Jaulmes et al. [2005], who do not make particular hypothesis on the evolution of the NSMDP. They build a learning algorithm for POMDPs solving, weighting recently experienced transitions more than older ones to account for the time dependency.

To plan robustly within an NSMDP, our approach consists in exploiting the slow LC evolution of the environment. Utilizing Lipschitz continuity to infer bounds on a function is common in the RL, bandit and optimization communities [Kleinberg et al., 2008, Rachelson and Lagoudakis, 2010, Pirotta et al., 2015, Pazis and Parr, 2013, Munos, 2014]. We implement this approach with a minimax-like algorithm [Fudenberg and Tirole, 1991], where the environment is seen as an adversarial agent.

## 4 Worst-case approach

We consider finding an optimal policy within an LC-NSMDP under the non-episodic task hypothesis. The latter prevents us from learning from previous experience data since they become outdated with time and no information samples have been collected yet for future time steps. An alternative is to use model-based $\mathbf { R I }$ algorithms such as MCTS. For a current state $s _ { 0 } ,$ such algorithms focus on finding the optimal action $a _ { 0 } ^ { * }$ by using a generative model. This action is then undertaken and the operation repeated at the next state. However, using the true NSMDP model for this purpose is an unrealistic hypothesis, since this model is generally unknown. We assume the agent does not have access to the true NSMDP model; instead, we introduce the notion of snapshot model.Intuitively, the snapshot associated to time $t _ { 0 }$ is a temporal slice of the NSMDP at $t _ { 0 }$

Definition 5. Snapshot of an NSMDP. The snapshot of an NSMDP $\{ S , T , \mathcal { A } , ( p _ { t } ) _ { t \in \mathcal { T } } , ( r _ { t } ) _ { t \in \mathcal { T } } \}$ at decision epoch $t _ { 0 } ,$ , denoted by $M D P _ { t _ { 0 } } ,$ , is the stationary MDP defined by the 4-tuple $\{ S , { \mathcal { A } } , p _ { t _ { 0 } } , r _ { t _ { 0 } } \}$ where $p _ { t _ { 0 } } ( s ^ { \prime } \mid s , a )$ and $r _ { t _ { 0 } } ( s , a , s ^ { \prime } )$ are the transition and reward functions of the NSMDP at $t _ { 0 }$ .

Similarly to the NSMDP, this definition induces the existence of the snapshot expected reward $R _ { t _ { 0 } }$ defined by $R _ { t _ { 0 } } : s , a \mapsto \mathbb { E } _ { s ^ { \prime } \sim p _ { t _ { 0 } } ( \cdot \vert s , a ) } \left\{ r _ { t _ { 0 } } ( s , a , s ^ { \prime } ) \right\}$ . Notice that the snapshot $\mathrm { M D P } _ { t _ { 0 } }$ is stationary and coincides with the NSMDP only at $t _ { 0 }$ . Particularly, one can generate a trajectory $\{ s _ { 0 } , r _ { 0 } , \cdot \cdot \cdot , s _ { k } \}$ within an NSMDP using the sequence of snapshots $\{ \mathrm { M D P } _ { t _ { 0 } } , \cdot \cdot \cdot , \mathrm { M D P } _ { t _ { 0 } + k - 1 } \}$ as a model. Overall, the hypothesis of using snapshot models amounts to considering a planning agent only able to get the current stationary model of the environment. In real-world problems, predictions often are uncertain or hard to perform e.g. in the thermal soaring problem of a glider.

We consider a generic planning agent at $s _ { 0 } , t _ { 0 }$ , using $\mathrm { M D P } _ { t _ { 0 } }$ as a model of the NSMDP. By planning, we mean conducting a look-ahead search within the possible trajectories starting from $s _ { 0 } , t _ { 0 }$ given a model of the environment. The search allows in turn to identify an optimal action w.r.t. the model. This action is then undertaken and the agent jumps to the next state where the operation is repeated. The consequence of planning with $\mathrm { M D P } _ { t _ { 0 } }$ is that the estimated value of an $s , t$ pair is the value of the optimal policy of $\mathrm { M D P } _ { t _ { 0 } }$ , written $V _ { \mathrm { M D P } _ { t _ { 0 } } } ^ { * } \left( s \right)$ . The true optimal value of s at t within the NSMDP does not match this estimate because of the non-stationarity. The intuition we develop is that, given the slow evolution rate of the environment, for a state s seen at a future decision epoch during the search, we can predict a scope into which the transition and reward functions at s lie.

Property 2. Set of admissible snapshot models. Consider an $( L _ { p } , L _ { r } ) \ – \ L C \ – \ N S M D P , \ s , t , a \ \in$ $s \times \bar { \tau } \times \bar { A }$ . The transition and expected reward functions $( p _ { t } , R _ { t } )$ of the snapshot $M D P _ { t }$ respect

$$
(p _ {t}, R _ {t}) \in \Delta_ {t} := \mathcal {B} _ {W _ {1}} \left(p _ {t - 1} (\cdot \mid s, a), L _ {p}\right) \times \mathcal {B} _ {| \cdot |} \left(R _ {t - 1} (s, a), L _ {R}\right)
$$

where $L _ { R } = L _ { p } + L _ { r }$ and $\boldsymbol { B } _ { d } \left( \boldsymbol { c } , \boldsymbol { r } \right)$ denotes the ball of centre c, defined with metric d and radius r.

For a future prediction at s, t, we consider the question of using a better model than $p _ { t _ { 0 } } , R _ { t _ { 0 } }$ . The underlying evolution of the NSMDP being unknown, a desirable feature would be to use a model leading to a policy that is robust to every possible evolution. To that end, we propose to use the snapshots corresponding to the worst possible evolution scenario under the constraints of Property 2. We claim that such a practice is an efficient way to 1) ensure robust performance to all possible evolutions of the NSMDP and 2) avoid catastrophic terminal states. Practically, this boils down to using a different value estimate for s at t than $V _ { \mathrm { M D P } _ { t _ { 0 } } } ^ { * } \left( s \right)$ which provided no robustness guarantees.

Given a policy ${ \pi } = ( { \pi } _ { t } ) _ { t \in { \mathcal { T } } }$ and a decision epoch $t ,$ a worst-case NSMDP corresponds to a sequence of transition and reward models minimizing the expected value of applying π in any pair $( s , t )$ , while remaining within the bounds of Property 2. We write $\overline { { V } } _ { t } ^ { \pi } ( s )$ this value for s at decision epoch t.

<table><tr><td>ε</td><td></td><td>RATS</td><td>DP-snapshot</td><td>DP-NSMDP</td></tr><tr><td rowspan="2">0</td><td>E [Σ r]</td><td>-0.026</td><td>0.48</td><td>0.47</td></tr><tr><td>CVaR</td><td>-0.81</td><td>-0.90</td><td>-0.9</td></tr><tr><td rowspan="2">0.5</td><td>E [Σ r]</td><td>-0.032</td><td>-0.46</td><td>-0.077</td></tr><tr><td>CVaR</td><td>-0.81</td><td>-0.90</td><td>-0.81</td></tr><tr><td rowspan="2">1</td><td>E [Σ r]</td><td>0.67</td><td>-0.78</td><td>0.66</td></tr><tr><td>CVaR</td><td>0.095</td><td>-0.90</td><td>-0.033</td></tr></table>

(a) Tree structure, d<sub>max</sub> = 2, A = {a<sub>1</sub>, a<sub>2</sub>}.  
(b) Expected return E [P r] and CVaR at 5%.  
Figure 1: Tree structure and results from the Non-Stationary bridge experiment.

$$
\overline {{V}} _ {t} ^ {\pi} (s) := \min _ {(p _ {i}, R _ {i}) \in \Delta_ {i}, \forall i \in \mathcal {T}} \mathbb {E} \left[ \sum_ {i = t} ^ {\infty} \gamma^ {i - t} R _ {i} (s _ {i}, a _ {i}) \Big | _ {a _ {i} \sim \pi_ {i} (\cdot \mid s _ {i}), s _ {i + 1} \sim p _ {i} (\cdot \mid s _ {i}, a _ {i})} ^ {s _ {t} = s} \right]\tag{1}
$$

Intuitively, the worst-case NSMDP is a model of a non-stationary environment leading to the poorest possible performance for $\pi$ , while being an admissible evolution of $\mathrm { { \mathbf { M D P } } } _ { t }$ . Let us define $\overline { { Q } } _ { t } ^ { \pi } ( \bar { s } , a )$ as the worst-case Q-value for the pair (s, a) at decision epoch t:

$$
\overline {{Q}} _ {t} ^ {\pi} (s, a) := \min _ {(p, R) \in \Delta_ {t}} \underset {s ^ {\prime} \sim p} {\mathbb {E}} \left[ R (s, a) + \gamma \overline {{V}} _ {t + 1} ^ {\pi} (s ^ {\prime}) \right].\tag{2}
$$

## 5 Risk-Averse Tree-Search algorithm

The algorithm. Tree search algorithms within MDPs have been well studied and cover two classes of search trees, namely closed loop [Keller and Helmert, 2013, Kocsis and Szepesvári, 2006, Browne et al., 2012] and open loop [Bubeck and Munos, 2010, Lecarpentier et al., 2018]. Following [Keller and Helmert, 2013], we consider closed loop search trees, composed of decision nodes alternating with chance nodes. We adapt their formulation to take time into account, resulting in the following definitions. A decision node at depth $t ,$ denoted by $\nu ^ { s , t }$ , is labeled by a unique state / decision epoch pair $( s , t )$ . The edges leading to its children chance nodes correspond to the available actions at $( s , t )$ A chance node, denoted by $\nu ^ { s , t , a }$ , is labeled by a state / decision epoch / action triplet $( s , t , a )$ . The edges leading to its children decision nodes correspond to the reachable state / decision epoch pairs $( s ^ { \prime } , t ^ { \prime } )$ after performing a in $( s , t )$ as illustrated by Figure 1a. We consider the problem of estimating the optimal action $a _ { 0 } ^ { * }$ at $s _ { 0 } , t _ { 0 }$ within a worst-case NSMDP, knowing $\mathbf { M D P } _ { t _ { 0 } }$ This problem is twofold. It requires 1) to estimate the worst-case NSMDP given $\mathrm { M D P } _ { t _ { 0 } }$ and 2) to explore the latter in order to identify $a _ { 0 } ^ { * }$ . We propose to tackle both problems with an algorithm inspired by the minimax algorithm [Fudenberg and Tirole, 1991] where the max operator corresponds to the agent’s policy, seeking to maximize the return; and the min operator corresponds to the worst-case model, seeking to minimize the return. Estimating the worst-case NSMDP requires to estimate the sequence of subsequent snapshots minimizing Equation 2. The inter-dependence of those snapshots (Equation 1) makes the problem hard to solve [Iyengar, 2005], particularly because of the combinatorial nature of the opponent’s action space. Instead, we propose to solve a relaxation of this problem, by considering snapshots only constrained by $\mathrm { M D P } _ { t _ { 0 } } ^ { \overline { { } } }$ . Making this approximation leaves a possibility to violate property 2 but allows for an efficient search within the developed tree and (as will be shown experimentally) leads to robust policies. For that purpose, we define the set of admissible snapshot models w.r.t. MDP<sub>t</sub> by $\Delta _ { t _ { 0 } } ^ { t } : = \mathcal { B } _ { W _ { 1 } } \left( p _ { t _ { 0 } } ( \cdot \vert s , \bar { a } ) , \bar { L } _ { p } \vert t - t _ { 0 } \vert \right) \times \mathcal { B } _ { \vert \cdot \vert } \left( R _ { t _ { 0 } } ( s , a ) , L _ { R } \vert t - t _ { 0 } \vert \right)$ . The relaxed analogues of Equations 1 and 2 for $s , t , a \in S \times \mathcal { T } \times \mathcal { A }$ are defined as follows:

$$
\begin{array}{r} \hat {V} _ {t _ {0}, t} ^ {\pi} (s) := \min _ {(p _ {i}, R _ {i}) \in \Delta_ {t _ {0}} ^ {i}, \forall i \in \mathcal {T}} \mathbb {E} \left[ \sum_ {i = t} ^ {\infty} \gamma^ {i - t} R _ {i} (s _ {i}, a _ {i}) \Big | _ {a _ {i} \sim \pi_ {i} (\cdot \mid s _ {i}), s _ {i + 1} \sim p _ {i} (\cdot \mid s _ {i}, a _ {i})} ^ {s _ {t} = s} \right], \\ \hat {Q} _ {t _ {0}, t} ^ {\pi} (s, a) := \min _ {(p, R) \in \Delta_ {t _ {0}} ^ {t}} \underset {s ^ {\prime} \sim p} {\mathbb {E}} \left[ R (s, a) + \gamma \hat {V} _ {t _ {0}, t + 1} ^ {\pi} (s ^ {\prime}) \right]. \end{array}
$$

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1: RATS algorithm
RATS ($s_0$, $t_0$, maxDepth)
$\nu_0 = \text{rootNode}(s_0, t_0)$ $\text{Minimax}(\nu_0)$ $\nu^* = \arg \max_{\nu'} \text{in}_{\nu_0.\text{children}} \nu'$.value
return $\nu^*$.action

Minimax ($\nu$, maxDepth)
if $\nu$ is DecisionNode then
    if $\nu.\text{state is terminal or } \nu.\text{depth} = \text{maxDepth}$ then
        return $\nu.\text{value} = \text{heuristicValue}(\nu.\text{state})$
    else
        return $\nu.\text{value} = \max_{\nu' \in \nu.\text{children}} \text{Minimax}(\nu', \text{maxDepth})$

else
    return $\nu.\text{value} = \min_{(p,R) \in \Delta_{t_0}^t} R(\nu) + \gamma \sum_{\nu' \in \nu.\text{children}} p(\nu' | \nu)\text{Minimax}(\nu', \text{maxDepth})$
</div>

Their optimal counterparts, while seeking to find the optimal policy, verify the following equations:

$$
\hat {V} _ {t _ {0}, t} ^ {*} (s) = \max _ {a \in \mathcal {A}} \hat {Q} _ {t _ {0}, t} ^ {*} (s, a),\tag{3}
$$

$$
\hat {Q} _ {t _ {0}, t} ^ {*} (s, a) = \min _ {(p, R) \in \Delta_ {t _ {0}} ^ {t}} \underset {s ^ {\prime} \sim p} {\mathbb {E}} \left[ R (s, a) + \gamma \hat {V} _ {t _ {0}, t + 1} ^ {*} (s ^ {\prime}) \right].\tag{4}
$$

We now provide a method to calculate those quantities within the nodes of the tree search algorithm. Max nodes. A decision node $\nu ^ { s , t }$ corresponds to a max node due to the greediness of the agent w.r.t. the subsequent values of the children. We aim at maximizing the return while retaining a risk-averse behavior. As a result, the value of $\nu ^ { s , t }$ follows Equation 3 and is defined as:

$$
V (\nu^ {s, t}) = \max _ {a \in \mathcal {A}} V (\nu^ {s, t, a}).\tag{5}
$$

Min nodes. A chance node $\nu ^ { s , t , a }$ corresponds to a min node due to the use of a worst-case NSMDP as a model which minimizes the value of $\nu ^ { s , t , a }$ w.r.t. the reward and the subsequent values of its children. Writing the value of $\nu ^ { s , t , a }$ as the value of $s , t , a$ , within the worst-case snapshot minimizing Equation 4, and using the children’s values as values for the next reachable states, leads to Equation 6.

$$
V (\nu^ {s, t, a}) = \min _ {(p, R) \in \Delta_ {t _ {0}} ^ {t}} R (s, a) + \gamma \underset {s ^ {\prime} \sim p} {\mathbb {E}} V (\nu^ {s ^ {\prime}, t + 1})\tag{6}
$$

Our approach considers the environment as an adversarial agent, as in an asymmetric two-player game, in order to search for a robust plan. The resulting algorithm, RATS for Risk-Averse Tree-Search, is described in Algorithm 1. Given an initial state / decision epoch pair, a minimax tree is built using the snapshot $\mathrm { M D P } _ { t _ { 0 } }$ and the operators corresponding to Equations 5 and 6 in order to estimate the worst-case snapshots at each depth. The tree is built, the action leading to the best possible value from the root node is selected and a real transition is performed. The next state is then reached, the new snapshot model $\mathrm { M D P } _ { t _ { 0 } + 1 }$ is acquired and the process re-starts. Notice the use of $R ( \nu )$ and $p ( \nu ^ { \prime } \mid \nu )$ in the pseudo-code: they are light notations respectively standing for $R _ { t } ( s , a )$ corresponding to a chance node $\nu \equiv \nu ^ { s , t , a }$ and the probability $p _ { t } ( s ^ { \prime } | s , a )$ to jump to a decision node $\nu ^ { \prime } \equiv \nu ^ { s ^ { \prime } , t + 1 }$ given a chance node $\nu \equiv \nu ^ { s , t , a }$ The tree built by RATS is entirely developed until the maximum depth $d _ { \operatorname* { m a x } } . \mathbf { A }$ heuristic function is used to evaluate the leaf nodes of the tree.

Analysis of RATS. We are interested in characterizing Algorithm 1 without function approximation and therefore will consider finite, countable, $\boldsymbol { s } \times \boldsymbol { A }$ sets. We now detail the computation of the min operator (Property 3), the computational complexity of RATS (Property 4) and the heuristic function.

Property 3. Closed-form expression of the worst case snapshot of a chance node. Following Algorithm 1, a solution to Equation 6 is given by:

$$
\hat {R} (s, a) = R _ {t _ {0}} (s, a) - L _ {R} | t - t _ {0} | \quad a n d \quad \hat {p} (\cdot \mid s, a) = (1 - \lambda) p _ {t _ {0}} (\cdot \mid s, a) + \lambda p _ {s a t} (\cdot \mid s, a)
$$

with $p _ { s a t } ( \cdot , \ | s , a ) = ( 0 , \cdot \cdot \cdot , 0 , 1 , 0 , \cdot \cdot \cdot , 0 )$ with 1 at position arg mi $\iota _ { s ^ { \prime } } V ( \nu ^ { s ^ { \prime } , t + 1 } ) , \lambda \ =$ $1 i f W _ { 1 } ( p _ { s a t } , p _ { 0 } ) \leq L _ { p } | t - t _ { 0 } | a n d \lambda = L _ { p } | t - t _ { 0 } | / W _ { 1 } ( p _ { s a t } , p _ { 0 } )$ otherwise.

Property 4. Computational complexity. The total computation complexity of Algorithm 1 is $\mathcal { O } ( \bar { B } | S | ^ { 1 . 5 } | A | ( | S | \bar { | A | } ) ^ { d _ { \operatorname* { m a x } } } )$ with B the number of time steps and $d _ { \mathrm { m a x } }$ the maximum depth.

Heuristic function. As in vanilla minimax algorithms, Algorithm 1 bootstraps the values of the leaf nodes with a heuristic function if these leaves do not correspond to terminal states. Given such a leaf node $\nu ^ { s , t }$ , a heuristic aims at estimating the value of the optimal policy at $( s , t )$ within the worst-case NSMDP, i.e. $\hat { V } _ { t _ { 0 } , t } ^ { * } ( s )$ . Let $H ( s , t )$ be such a heuristic function, we call heuristic error in $( s , t )$ the difference between $H ( s , t )$ and $\hat { V } _ { t _ { 0 } , t } ^ { * } ( s )$ . Assuming that the heuristic error is uniformly bounded, the following property provides an upper bound on the propagated error due to the choice of H.

Property 5. Upper bound on the propagated heuristic error within RATS. Consider an agent executing Algorithm 1 at $s _ { 0 } , t _ { 0 }$ with a heuristic function H. We note the set of all leaf nodes. Suppose that the heuristic error is uniformly bounded, i.e. $\exists \delta > 0 , \forall \nu ^ { s , t } \in \mathcal { L } , | H ( s ) - \hat { V } _ { t _ { 0 } , t } ^ { * } ( s ) | \le \delta .$ Then we have for every decision and chance nodes $\nu ^ { s , t }$ and $\nu ^ { s , t , a }$ , at any depth $d \in [ 0 , d _ { \operatorname* { m a x } } ] \colon$

$$
| V (\nu^ {s, t}) - \hat {V} _ {t _ {0}, t} ^ {*} (s) | \leq \gamma^ {(d _ {\max} - d)} \delta \quad a n d \quad | V (\nu^ {s, t, a}) - \hat {Q} _ {t _ {0}, t} ^ {*} (s, a) | \leq \gamma^ {(d _ {\max} - d)} \delta .
$$

This last result implies that with any heuristic function H inducing a uniform heuristic error, the propagated error at the root of the tree is guaranteed to be upper bounded by $\gamma ^ { d _ { \operatorname* { m a x } } } \delta$ . In particular, since the reward function is bounded by hypothesis, we have $\hat { V } _ { t _ { 0 } , t } ^ { * } ( s ) \leq 1 / ( 1 - \gamma )$ . Thus, selecting for instance the zero function ensures a root node heuristic error of at most $\gamma ^ { d _ { \operatorname* { m a x } } } / ( 1 - \gamma )$ . In order to improve the precision of the algorithm, we propose to guide the heuristic by using a function reflecting better the value of state s at leaf node $\nu ^ { s , t }$ . The ideal function would of course be $H ( s ) = \hat { V } _ { t _ { 0 } , t } ^ { * } ( s )$ reducing the heuristic error to zero, but this is intractable. Instead, we suggest to use the value of s within the snapshot MDP<sub>t</sub> using an evaluation policy π, i.e. $H ( s ) = V _ { \mathrm { M D P } _ { t } } ^ { \pi } \bar { ( } s )$ . This snapshot is also not available, but Property 6 provides a range wherein this value lies.

Property 6. Bounds on the snapshots values. Let $s \in S ,$ , π a stationary policy, $M D P _ { t _ { 0 } }$ and $M D P _ { t }$ two snapshot MDPs, $t , t _ { 0 } \in \mathcal { T } ^ { 2 }$ be. We note $V _ { M D P _ { i } } ^ { \pi } ( s )$ the value of s within MDP following π. Then,

$$
| V _ {M D P _ {t _ {0}}} ^ {\pi} (s) - V _ {M D P _ {t}} ^ {\pi} (s) | \leq | t - t _ {0} | L _ {R} / (1 - \gamma).
$$

Since $\mathrm { M D P } _ { t _ { 0 } }$ is available, $V _ { \mathrm { M D P } _ { t _ { 0 } } } ^ { \pi } \left( s \right)$ can be estimated, e.g. via Monte-Carlo roll-outs. Let $\widehat { V } _ { \mathrm { M D P } _ { t _ { 0 } } } ^ { \pi } \left( s \right)$ denote such an estimate. Following Property $6 , V _ { \mathrm { M D P } _ { t _ { 0 } } } ^ { \pi } ( s ) - | t - t _ { 0 } | L _ { R } / ( 1 - \gamma ) \le V _ { \mathrm { M D P } _ { t } } ^ { \pi } ( s )$ . Hence, 0 a worst-case heuristic on $V _ { \mathrm { M D P } _ { t } } ^ { \pi } ( s )$ is $H ( s ) = \widehat { V } _ { \mathrm { M D P } _ { t _ { 0 } } } ^ { \pi } ( s ) - \vert t - t _ { 0 } \vert L _ { R } / ( 1 - \gamma )$ . The bounds provided by Property 5 decrease quickly with $d _ { \operatorname* { m a x } } .$ , and given that $d _ { \mathrm { m a x } }$ is large enough, RATS provides the optimal risk-averse maximizing the worst-case value for any evolution of the NSMDP.

## 6 Experiments

We compare the RATS algorithm with two policies <sup>1</sup>. The first one, named DP-snapshot, uses Dynamic Programming to compute the optimal actions w.r.t. the snapshot models at each decision epoch. The second one, named DP-NSMDP, uses the real NSMDP as a model to provide its optimal action. The latter behaves as an omniscient agent and should be seen as an upper bound on the performance. We choose a particular grid-world domain coined “Non-Stationary bridge” illustrated in Appendix, Section 7. An agent starts at the state labeled S in the center and the goal is to reach one of the two terminal states labeled G where a reward of +1 is received. The gray cells represent holes that are terminal states where a reward of -1 is received. Reaching the goal on the right leads to the highest payoff since it is closest to the initial state and a discount factor $\gamma = 0 . 9$ is applied. The actions are $\dot { \mathcal { A } } = \{ \mathrm { U p } \}$ , Right, Down, Left . The transition function is stochastic and non-stationary. At decision epoch $t = 0$ , any action deterministically yields the intuitive outcome. With time, when applying Left or Right, the probability to reach the positions usually stemming from Up and Down increases symmetrically until reaching 0.45. We set the Lipschitz constant $L _ { p } = 1$ . Aside, we introduce a parameter $\epsilon \in [ 0 , 1 ]$ controlling the behavior of the environment. $\bar { \mathrm { H } ^ { \prime } } \epsilon = 0$ , only the left-hand side bridge becomes slippery with time. It reflects a close to worst-case evolution for a policy aiming to the left-hand side goal. $\mathrm { I f } \epsilon = 1$ , only the right-hand side bridge becomes slippery with time. It reflects a close to worst-case evolution for a policy aiming to the right-hand side goal. In between, the misstep probability is proportionally balanced between left and right. One should note that changing  from 0 to 1 does not cover all the possible evolutions from $\mathrm { M D P } _ { t _ { 0 } }$ but provides a concrete, graphical illustration of RATS’s behavior for various possible evolutions of the NSMDP.

![](images/2773f264958aeb745275ab3ab817915b02a643342a32f02c14e4699a313c1e05.jpg)  
(a) Discounted return vs , 50% of standard deviation.

![](images/cd39d156526226c7538f14306314f6c3f3807bb8d8f7110ae0a026673e723832.jpg)  
(b) Discounted return distributions $\epsilon \in \{ 0 , 0 . 5 , 1 \}$  
Figure 2: Discounted return of the three algorithms for various values of .

We tested RATS with $d _ { \operatorname* { m a x } } = 6$ so that leaf nodes in the search tree are terminal states. Hence, the optimal risk-averse policy is applied and no heuristic approximation is made. Our goal is to demonstrate that planning in this worst-case NSMDP allows to minimize the loss given any possible evolution of the environment. To illustrate this, we report results reflecting different evolutions of the same NSMDP using the  factor. It should be noted that, at $t = 0 ,$ , RATS always moves to the left, even if the goal is further, since going to the right may be risky if the probabilities to go Up and Down increase. This corresponds to the careful, risk-averse, behavior. Conversely, DP-snapshot always moves to the right since $\mathrm { M D P _ { 0 } }$ does not capture this risk. As a result, the $\epsilon = 0$ case reflects a favorable evolution for DP-snapshot and a bad one for RATS. The opposite occurs with $\epsilon = 1$ where the cautious behavior dominates over the risky one, and the in-between cases mitigate this effect.

In Figure 2a, we display the achieved expected return for each algorithm as a function of , i.e. as a function of the possible evolutions of the NSMDP. As expected, the performance of DP-snapshot strongly depends on this evolution. It achieves high return for $\epsilon = 0$ and low return for $\epsilon = 1$ Conversely, the performance of RATS varies less across the different values of . The effect illustrated here is that RATS maximizes the minimal possible return given any evolution of the NSMDP. It provides the guarantee to achieve the best return in the worst-case. This behavior is highly desirable when one requires robust performance guarantees as, for instance, in critical certification processes. Figure 2b displays the return distributions of the three algorithms for $\epsilon \in \{ 0 , 0 . 5 , 1 \}$ . The effect seen here is the tendency for RATS to diminish the left tail of the distribution corresponding to low returns for each evolution. It corresponds to the optimized criteria, i.e. robustly maximizing the worst-case value. A common risk measure is the Conditional Value at Risk (CVaR) defined as the expected return in the worst $q \%$ cases. We illustrate the CVaR at 5% achieved by each algorithm in Table 1b. Notice that RATS always maximizes the CVaR compared to both DP-snapshot and DP-NSMDP. Indeed, even if the latter uses the true model, the optimized criteria in DP is the expected return.

## 7 Conclusion

We proposed an approach for robust planning in non-stationary stochastic environments. We introduced the framework of Lipchitz Continuous Non-Stationary MDPs (NSMDPs) and derived the Risk-Averse Tree-Search (RATS) algorithm, to predict the worst-case evolution and to plan optimally w.r.t. this worst-case NSMDP. We analyzed RATS theoretically and showed that it approximates a worst-case NSMDP with a control parameter that is the depth of the search tree. We showed empirically the benefit of the approach that searches for the highest lower bound on the worst achievable score. RATS is robust to every possible evolution of the environment, i.e. maximizing the expected worst-case outcome on the whole set of possible NSMDPs. Our method was applied to the uncertainty on the evolution of a model. Generally, it could be extended to any uncertainty on the model used for planning, given bounds on the set of the feasible models. The purpose of this contribution is to lay a basis of worst-case analysis for robust solutions to NSMDPs. As is, RATS is computationally intensive and scaling the algorithm to larger problems is an exciting future challenge.

## Acknowledgments

This research was supported by the Occitanie region, France.

## References

Y. Abbasi, P. L. Bartlett, V. Kanade, Y. Seldin, and C. Szepesvári. Online learning in Markov decision processes with adversarially chosen transition probability distributions. In Advances in Neural Information Processing Systems, pages 2508–2516, 2013.

D. Abel, D. Arumugam, L. Lehnert, and M. Littman. State Abstractions for Lifelong Reinforcement Learning. In International Conference on Machine Learning, pages 10–19, 2018a.

D. Abel, Y. Jinnai, S. Y. Guo, G. Konidaris, and M. Littman. Policy and Value Transfer in Lifelong Reinforcement Learning. In International Conference on Machine Learning, pages 20–29, 2018b.

K. Asadi, D. Misra, and M. L. Littman. Lipschitz continuity in model-based reinforcement learning. arXiv preprint arXiv:1804.07193, 2018.

C. B. Browne, E. Powley, D. Whitehouse, S. M. Lucas, P. I. Cowling, P. Rohlfshagen, S. Tavener, D. Perez, S. Samothrakis, and S. Colton. A survey of Monte Carlo tree search methods. IEEE Transactions on Computational Intelligence and AI in games, 4(1):1–43, 2012.

S. Bubeck and R. Munos. Open loop optimistic planning. In 10th Conference on Learning Theory, 2010.

L. Campo, P. Mookerjee, and Y. Bar-Shalom. State estimation for systems with sojourn-timedependent Markov model switching. IEEE Transactions on Automatic Control, 36(2):238–243, 1991.

S. P. Choi, D.-y. Yeung, and N. L. Zhang. Hidden-mode Markov decision processes. In IJCAI Workshop on Neural, Symbolic, and Reinforcement Methods for Sequence Learning. Citeseer, 1999.

S. P. Choi, D.-Y. Yeung, and N. L. Zhang. Hidden-mode Markov decision processes for nonstationary sequential decision making. In Sequence Learning, pages 264–287. Springer, 2000.

S. P. Choi, N. L. Zhang, and D.-Y. Yeung. Solving hidden-mode Markov decision problems. In Proceedings of the 8th International Workshop on Artificial Intelligence and Statistics, Key West, Florida, USA, 2001.

J. J. Chung, N. R. Lawrance, and S. Sukkarieh. Learning to soar: Resource-constrained exploration in reinforcement learning. The International Journal of Robotics Research, 34(2):158–172, 2015.

B. C. Csáji and L. Monostori. Value function based reinforcement learning in changing Markovian environments. Journal of Machine Learning Research, 9(Aug):1679–1709, 2008.

B. C. Da Silva, E. W. Basso, A. L. Bazzan, and P. M. Engel. Dealing with non-stationary environments using context detection. In Proceedings of the 23rd International Conference on Machine Learning, pages 217–224. ACM, 2006.

W. Dabney, M. Rowland, M. G. Bellemare, and R. Munos. Distributional reinforcement learning with quantile regression. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.

T. Dick, A. Gyorgy, and C. Szepesvari. Online learning in Markov decision processes with changing cost sequences. In International Conference on Machine Learning, pages 512–520, 2014.

K. Doya, K. Samejima, K.-i. Katagiri, and M. Kawato. Multiple model-based reinforcement learning. Neural computation, 14(6):1347–1369, 2002.

E. Even-Dar, S. M. Kakade, and Y. Mansour. Online Markov Decision Processes. Mathematics of Operations Research, 34(3):726–736, 2009.

D. Fudenberg and J. Tirole. Game theory. Cambridge, Massachusetts, 393(12):80, 1991.

E. Hadoux. Markovian sequential decision-making in non-stationary environments: application to argumentative debates. PhD thesis, UPMC, Sorbonne Universités CNRS, 2015.

E. Hadoux, A. Beynier, and P. Weng. Sequential decision-making under non-stationary environments via sequential change-point detection. In Learning over Multiple Contexts (LMCE), 2014.

G. N. Iyengar. Robust dynamic programming. Mathematics of Operations Research, 30(2):257–280, 2005.

R. Jaulmes, J. Pineau, and D. Precup. Learning in non-stationary partially observable Markov decision processes. In ECML Workshop on Reinforcement Learning in non-stationary environments, volume 25, pages 26–32, 2005.

L. P. Kaelbling, M. L. Littman, and A. R. Cassandra. Planning and acting in partially observable stochastic domains. Artificial intelligence, 101(1-2):99–134, 1998.

T. Keller and M. Helmert. Trial-based heuristic tree search for finite horizon MDPs. In ICAPS, 2013.

R. Kleinberg, A. Slivkins, and E. Upfal. Multi-armed bandits in metric spaces. In Proceedings of the fortieth annual ACM symposium on Theory of computing, pages 681–690. ACM, 2008.

L. Kocsis and C. Szepesvári. Bandit based Monte-Carlo planning. In European conference on machine learning, pages 282–293. Springer, 2006.

E. Lecarpentier, S. Rapp, M. Melo, and E. Rachelson. Empirical evaluation of a Q-Learning Algorithm for Model-free Autonomous Soaring. arXiv preprint arXiv:1707.05668, 2017.

E. Lecarpentier, G. Infantes, C. Lesire, and E. Rachelson. Open loop execution of tree-search algorithms. IJCAI, 2018.

S. H. Lim, H. Xu, and S. Mannor. Reinforcement learning in robust markov decision processes. In Advances in Neural Information Processing Systems, pages 701–709, 2013.

R. Munos. From bandits to monte-carlo tree search: The optimistic principle applied to optimization and planning. Foundations and Trends R in Machine Learning, 7(1):1–129, 2014.

J. Pazis and R. Parr. PAC Optimal Exploration in Continuous Space Markov Decision Processes. In AAAI, 2013.

M. Pirotta, M. Restelli, and L. Bascetta. Policy gradient in lipschitz Markov Decision Processes. Machine Learning, 100(2-3):255–283, 2015.

M. L. Puterman. Markov decision processes: discrete stochastic dynamic programming. John Wiley & Sons, 2014.

E. Rachelson and M. G. Lagoudakis. On the locality of action domination in sequential decision making. 2010.

D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. Van Den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, et al. Mastering the game of Go with deep neural networks and tree search. Nature, 529(7587):484, 2016.

D. L. Silver, Q. Yang, and L. Li. Lifelong Machine Learning Systems: Beyond Learning Algorithms. In AAAI Spring Symposium: Lifelong Machine Learning, volume 13, page 05, 2013.

R. S. Sutton, A. G. Barto, et al. Reinforcement learning: An introduction. MIT press, 1998.

I. Szita, B. Takács, and A. Lörincz. ε-mdps: Learning in varying environments. Journal of Machine Learning Research, 3(Aug):145–174, 2002.

C. Villani. Optimal transport: old and new, volume 338. Springer Science & Business Media, 2008.

M. A. Wiering. Reinforcement learning in dynamic environments using instantiated information. In Machine Learning: Proceedings of the Eighteenth International Conference (ICML2001), pages 585–592, 2001.

# Non-Stationary Markov Decision Processes a Worst-Case Approach using Model-Based Reinforcement Learning

Appendix

Erwan Lecarpentier Université de Toulouse ONERA - The French Aerospace Lab erwan.lecarpentier@isae-supaero.fr

Emmanuel Rachelson Université de Toulouse ISAE-SUPAERO emmanuel.rachelson@isae-supaero.fr

In the following proofs, the dual formulation of the 1-Wasserstein distance is used several times. We include the definition here for reference purpose.

Definition 1. Dual formulation of the 1-Wasserstein distance. Let $( X , d _ { X } )$ be a Polish metric space and $\mu , \nu$ any two probability measures on X. The dual formulation of the 1-Wasserstein distance between µ and ν is defined by

$$
W _ {1} (\mu , \nu) = \sup _ {f \in L i p _ {1}} \int_ {X} f (x) d (\mu - \nu) (x)\tag{1}
$$

where $L i p _ { 1 }$ denotes the set of the continuous mappings $X \to \mathbb { R }$ with a minimal Lipschitz constant bounded $b y 1$

## 1 Proof of Property 1

Consider an $( L _ { p } , L _ { r } ) \mathrm { - L C - N S M D P }$ . Let $s , t , a , \hat { t } \in \mathcal { S } \times \mathcal { T } \times \mathcal { A } \times \mathcal { T }$ be. By definition of the expected reward function, the following holds:

$$
\begin{array}{l} R _ {t} (s, a) - R _ {\hat {t}} (s, a) = \int_ {\mathcal {S}} \Big (p _ {t} (s ^ {\prime} \mid s, a) r _ {t} (s, a, s ^ {\prime}) - p _ {\hat {t}} (s ^ {\prime} \mid s, a) r _ {\hat {t}} (s, a, s ^ {\prime}) \Big) d s ^ {\prime} \\ \qquad = \int_ {\mathcal {S}} \Big (r _ {t} (s, a, s ^ {\prime}) \Big [ p _ {t} (s ^ {\prime} \mid s, a) - p _ {\hat {t}} (s ^ {\prime} \mid s, a) \Big ] \\ \qquad \qquad + p _ {\hat {t}} (s ^ {\prime} \mid s, a) \Big [ r _ {t} (s, a, s ^ {\prime}) - r _ {\hat {t}} (s, a, s ^ {\prime}) \Big ] \Big) d s ^ {\prime} \\ \qquad = \int_ {\mathcal {S}} r _ {t} (s, a, s ^ {\prime}) \Big [ p _ {t} (s ^ {\prime} \mid s, a) - p _ {\hat {t}} (s ^ {\prime} \mid s, a) \Big ] d s ^ {\prime} \\ \qquad + \int_ {\mathcal {S}} p _ {\hat {t}} (s ^ {\prime} \mid s, a) \Big [ r _ {t} (s, a, s ^ {\prime}) - r _ {\hat {t}} (s, a, s ^ {\prime}) \Big ] d s ^ {\prime} \\ \qquad \leq \sup _ {\| f \| _ {L} \leq 1} \int_ {\mathcal {S}} f (s ^ {\prime}, t ^ {\prime}) \Big [ p _ {t} (s ^ {\prime} \mid s, a) - p _ {\hat {t}} (s ^ {\prime} \mid s, a) \Big ] d s ^ {\prime} \\ \qquad + \int_ {\mathcal {S}} p _ {\hat {t}} (s ^ {\prime} \mid s, a) L _ {r} | t - \hat {t} | d s ^ {\prime} \\ \qquad \leq W _ {1} (p (\cdot \mid s, t, a), p (\cdot \mid s, \hat {t}, a)) + L _ {r} | t - \hat {t} | \\ \qquad \leq (L _ {p} + L _ {r}) | t - \hat {t} | \end{array}
$$

33rd Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada.

Where we used the triangle inequality, the fact that r is a bounded function and the dual formulation of the 1-Wasserstein distance (see Definition 1). The same inequality can be derived with the opposite terms which concludes the proof by taking the absolute value.

## 2 Proof of Property 2

Proof. The proof is straightforward using the Lipschitz property of Definition 4 and Property 1.

## 3 Proof of Property 4

Let us first calculate the cost of constructing a tree with the minimax procedure. Following Algorithm 1, a tree is composed of at most $n _ { l }$ leaf nodes, $n _ { d }$ non-leaf decision nodes and $n _ { c }$ chance nodes, with the following values for the integers $n _ { l } , n _ { d }$ and $n _ { c } \colon$

$$
n _ {l} = \left(| \mathcal {S} | | \mathcal {A} |\right) ^ {d _ {\max}}, n _ {d} = \sum_ {i = 0} ^ {d _ {\max} - 1} \left(| \mathcal {S} | | \mathcal {A} |\right) ^ {i}, \text {and} n _ {c} = | \mathcal {A} | B.
$$

As a result, we have that $n _ { l }$ is $O ( ( | S | | A | ) ^ { d _ { \operatorname* { m a x } } } )$ $n _ { d }$ is $\mathcal { O } ( ( \vert \boldsymbol { S } \vert \vert \mathcal { A } \vert ) ^ { d _ { \operatorname* { m a x } } - 1 } )$ and $n _ { c }$ is $\mathcal { O } ( \vert \mathcal { A } \vert ( \vert \mathcal { S } \vert \vert \mathcal { A } \vert ) ^ { d _ { \operatorname* { m a x } } - 1 } )$ . We note respectively $c _ { l } , \ c _ { d }$ and $c _ { c }$ the number of operations required to compute the values of a leaf node, a non-leaf decision node and a chance node. To compute the whole tree we need to build and evaluate all the nodes, resulting in at most the following number of operations:

$$
n _ {l} c _ {l} \times n _ {d} c _ {d} \times n _ {c} c _ {c}.\tag{2}
$$

We will assume that $c _ { l }$ is $\mathcal { O } ( 1 )$ without further details on the nature of the heuristic function. As the value of a non-leaf decision node is computed by finding the maximum value among the children, we have that $c _ { d }$ is $\mathcal { O } ( | A | )$ . From Theorem 3, the evaluation of a chance node is equivalent to computing a 1-Wasserstein distance, which is a linear program. Following Vaidya’s algorithm [Vaidya, 1989], the cost in the worst-case is $\mathcal { O } ( | S | ^ { 2 . 5 } )$ where is the dimension of the problem in our case. As a result, $c _ { c }$ is $\mathcal { O } ( | S | ^ { 2 . 5 } )$ ). Replacing all the values in Equation 2, we deduce that the total number of operation of computing a tree is

$$
\mathcal {O} \left(| \mathcal {S} | ^ {1. 5} \left(| \mathcal {S} | | \mathcal {A} |\right) ^ {d _ {\max}}\right).
$$

After computing a tree, the action maximizing the value should be selected which has complexity $\mathcal { O } ( | A | )$ ). The operation being repeated for every time steps, one should multiply everything by $B _ { ; }$ the total number of time steps for which the algorithm is run. As a result, the total computational complexity of RATS is

$$
\mathcal {O} \left(B | \mathcal {S} | ^ {1. 5} | \mathcal {A} | \left(| \mathcal {S} | | \mathcal {A} |\right) ^ {d _ {\max}}\right).
$$

## 4 Proof of Property 3

We are looking for a closed-form expression of the value of a chance node $\nu ^ { s , t , a }$ as defined in Equation 6 recalled below.

$$
(\bar {p}, \bar {R}) = \underset {(p, R) \in \Delta_ {t _ {0}, t}} {\arg \min} R (s, a) + \gamma \mathbb {E} _ {s ^ {\prime} \sim p (\cdot | s, a)} V (\nu^ {s ^ {\prime}, t + 1})
$$

Obviously, we have that $\bar { R } = R _ { t _ { 0 } } ( s , a ) - L _ { R } | t - t _ { 0 } |$ and p¯ is given by:

$$
\bar {p} = \underset {p \in \mathcal {B} _ {W _ {1}} (p _ {t _ {0}} (\cdot | s, a), L _ {p} | t - t _ {0} |)} {\arg \min} \sum_ {s ^ {\prime}} p (s ^ {\prime} \mid s, a) V (\nu^ {s ^ {\prime}, t + 1})
$$

where $ { \boldsymbol { B } } _ { d } ( c , r )$ denotes the ball of center $c ,$ defined with metric d and radius $^ { r } \cdot$ Since we are in the discrete case, we enumerate through the elements of $s$ and write the vectors $p \equiv ( p ( s ^ { \prime } \mid s , a ) ) _ { s ^ { \prime } } ,$ $p _ { 0 } \equiv ( p _ { t _ { 0 } } ( s ^ { \prime } \mid s , a ) ) _ { s ^ { \prime } }$ and $v \equiv ( V ( \nu ^ { s ^ { \prime } , t + 1 } ) ) _ { s ^ { \prime } }$ . The problem can then be re-written as follows:

$$
\bar {p} = \underset {p} {\arg \min} \quad p ^ {\top} v\tag{3}
$$

$$
\mathrm{s.t.} p ^ {\top} \mathbf {1} = 1\tag{4}
$$

$$
p \geq 0\tag{5}
$$

$$
W _ {1} (p, p _ {0}) \leq C\tag{6}
$$

Where we have $\mathbf { 1 } \in \mathbb { R } ^ { | S | }$ a vector of ones, $C = L _ { p } | t - t _ { 0 } |$ and the 1-Wasserstein metric between two discrete distributions written in dual form following Lemma 1 as:

$$
W _ {1} (u, v) = \max _ {f} f ^ {\top} (u - v)\tag{7}
$$

$$
\text {s.t.} A f \leq b
$$

Where the matrix A and vector b are defined such that for any indexes $i , j$ we have $| f _ { i } - f _ { j } | \le d _ { i , j }$ with $d _ { i , j }$ the metric defined over the measured space, in our case the state space $s$ . Hence we propose to solve the program 3 under constraints 4 to 6. Let us first show that this problem is convex. Clearly, the objective function in Equation 3 is linear, hence convex, and the constraints 4 and 5 define a convex set. We prove that the 1-Wasserstein distance is convex in Lemma 1.

Lemma 1. Convexity of the 1-Wasserstein distance. The 1-Wasserstein distance is convex i.e. for $\lambda \in [ 0 , 1 ] , ( X , d _ { X } )$ a Polish space and any three probability measures $w _ { 0 } , w _ { 1 } , w _ { 2 }$ on $X ,$ , the following holds:

$$
W _ {1} (w _ {0}, \lambda w _ {1} + (1 - \lambda) w _ {2}) \leq \lambda W _ {1} (w _ {0}, w _ {1}) + (1 - \lambda) W _ {1} (w _ {0}, w _ {2})
$$

Proof. We use the dual representation of the 1-Wasserstein distance of Definition 1.

$$
\begin{array}{l} W _ {1} (w _ {0}, \lambda w _ {1} + (1 - \lambda) w _ {2}) \\ \quad = \sup _ {f \in \mathrm{Lip} _ {1}} \int_ {X} f (x) (w _ {0} (x) - \lambda w _ {1} (x) - (1 - \lambda) w _ {2} (x)) d x \\ \quad = \sup _ {f \in \mathrm{Lip} _ {1}} \int_ {X} (\lambda f (x) (w _ {0} (x) - w _ {1} (x)) + (1 - \lambda) f (x) (w _ {0} (x) - w _ {2} (x))) d x \\ \quad \leq \lambda \sup _ {f \in \mathrm{Lip} _ {1}} \int_ {X} f (x) (w _ {0} (x) - w _ {1} (x)) d x + (1 - \lambda) \sup _ {f \in \mathrm{Lip} _ {1}} \int_ {X} f (x) (w _ {0} (x) - w _ {2} (x)) d x \\ \quad \leq \lambda W _ {1} (w _ {0}, w _ {1}) + (1 - \lambda) W _ {1} (w _ {0}, w _ {2}) \end{array}
$$

Where we used the linearity of the integral and the triangle inequality on the sup operator. □

The program 3 is thus convex. One can also observe that the gradient of the objective function is constant, equal to +v. Furthermore, $p _ { 0 }$ is an admissible initial point that we could use for a gradient descent method. However, given $p _ { 0 }$ , following the descent direction v may break the constraints 4 and 5. One would have to project this gradient onto a certain, unknown, set of hyperplanes in order to apply the gradient method descent. Let us note proj(v) the resulting projected gradient, that is unknown.

We remark that the vector $p _ { \mathrm { s a t } } = ( 0 , \cdots , 0 , 1 , 0 , \cdots , 0 )$ with 1 at the index arg min ${ \mathrm { ~ \it ~ \ i ~ v ~ } } _ { i }$ where $v _ { i }$ denotes the ith coefficient of $v ,$ is the optimal solution of the program 3 when we remove the Wasserstein constraint 6. One can observe that the optimal solution with the constraint 6 would as well be $p _ { \mathrm { s a t } }$ if the constant C is $b i g$ enough. As a result, the descent direction $\nabla = p _ { \mathrm { s a t } } - p _ { 0 }$ is the one to be followed in this setting when applying the gradient descent method to this case. Furthermore, following  from $p _ { 0 }$ until $p _ { \mathrm { s a t } }$ never breaks the constraints 4 and 5. Since the gradient of the objective function is constant, there can exist only one proj(v). fulfils the requirements, hence we have $\mathrm { p r o j } ( v ) = \nabla$

We can now apply the gradient method descent with the following 1-shot rule since the gradient is constant:

$$
\bar {p} := p _ {0} + \lambda \nabla \text {with,} \left\{ \begin{array}{l} \lambda = 1 \text {if} W _ {1} (p _ {\mathrm{sat}}, p _ {0}) \leq C \\ \lambda = C / W _ {1} (p _ {\mathrm{sat}}, p _ {0}) \end{array} \right.
$$

Indeed, in the first case, we can follow until the extreme distribution $p _ { \mathrm { s a t } }$ without breaking the constraint 6. Going further is trivially infeasible.

In the second case, we have to stop in between so that the constraint 6 is saturated. In such a case, we cannot go further without breaking this constraint and we recall that no projected gradient could be found by uniqueness of this gradient in our setting. Hence we have the following equality:

$$
\begin{array}{c} W _ {1} (p _ {0} + \lambda \nabla , p _ {0}) = C \\ \underset {A f \leq b} {\max} f ^ {\top} (p _ {0} + \lambda \nabla - p _ {0}) = C \\ \lambda \underset {A f \leq b} {\max} f ^ {\top} \nabla = C \\ \lambda = C / W _ {1} (p _ {\mathrm{sat}}, p _ {0}) \end{array}
$$

Where we used the fact that $\nabla = p _ { \mathrm { s a t } } - p _ { 0 }$ . The latter result concludes the proof.

## 5 Proof of Property 5

Let us consider a tree developed with Algorithm 1 with a heuristic function $\begin{array} { r } { H : s \mapsto H ( s ) } \end{array}$ used to estimate the value of a leaf node. The set of the leaves nodes is denoted by and we have the following uniform upper bound $\delta > 0$ on the heuristic error:

$$
\forall \nu^ {s, t} \in \mathcal {L}, | H (s) - \overline {{V}} _ {t _ {0}, t} ^ {*} (s) | <   \delta\tag{8}
$$

We want to prove the following result for a decision and chance nodes $\nu ^ { s , t }$ and $\nu ^ { s , t , a }$ at any depth $d \in [ 0 , d _ { \operatorname* { m a x } } ] ;$

$$
| V (\nu^ {s, t}) - \overline {{V}} _ {t _ {0}, t} ^ {*} (s) | \leq \gamma^ {(d _ {\max} - d)} \delta\tag{9}
$$

$$
| V (\nu^ {s, t, a}) - \overline {{Q}} _ {t _ {0}, t} ^ {*} (s, a) | \leq \gamma^ {(d _ {\max} - d)} \delta\tag{10}
$$

The proof is made by induction, starting at depth $d _ { \mathrm { m a x } }$ and reversely ending at depth $0 . \mathrm { A t } d _ { \mathrm { m a x } }$ , the nodes are leaf nodes, their values is estimated with the heuristic function i.e. $V ( \nu ^ { s , t } ) = H ( s )$ . Hence the result is directly proven by hypothesis in Equation 8. We will now start by proving the result for the chance nodes which come as the first parents of the decision node for which we initialized the induction proof. Then we extend it to the parents decision nodes which completes the proof.

Chance nodes case. Consider any chance node $\nu ^ { s , t , a }$ at depth $d \in [ 0 , d _ { \operatorname* { m a x } } ]$ . We suppose that the property is true for depth $d + 1$ , thus we have for any decision node at $d + 1$ denoted by $\nu ^ { s ^ { \prime } , t ^ { \prime } }$

$$
| V (\nu^ {s ^ {\prime}, t ^ {\prime}}) - \overline {{V}} _ {t _ {0}, t ^ {\prime}} ^ {*} (s ^ {\prime}) | \leq \gamma^ {(d _ {\max} - (d + 1))} \delta
$$

Following Equation 6 of the paper, we have by construction:

$$
V (\nu^ {s, t, a}) = \overline {{R}} _ {t} (s, a) + \gamma \sum_ {s ^ {\prime}} \overline {{p}} _ {t} (s ^ {\prime} \mid s, a) V (\nu^ {s ^ {\prime}, t ^ {\prime}})
$$

By definition, the true Q-value function defined by the Bellman Equation 2 gives the true target value:

$$
\overline {{Q}} _ {t _ {0}, t} ^ {*} (s, a) = \overline {{R}} _ {t} (s, a) + \gamma \sum_ {s ^ {\prime}} \overline {{p}} _ {t} (s ^ {\prime} \mid s, a) \overline {{V}} _ {t _ {0}, t ^ {\prime}} ^ {*} (s ^ {\prime})
$$

Hence, using the induction hypothesis, we have the following inequalities proving the result of Equation 10:

$$
\begin{array}{r l} & {| V (\nu^ {s, t, a}) - \overline {{Q}} _ {t _ {0}, t} ^ {*} (s, a) | = \gamma \left| \sum_ {s ^ {\prime}} \overline {{p}} _ {t} (s ^ {\prime} \mid s, a) V (\nu^ {s ^ {\prime}, t ^ {\prime}}) - \sum_ {s ^ {\prime}} \overline {{p}} _ {t} (s ^ {\prime} \mid s, a) \overline {{V}} _ {t _ {0}, t ^ {\prime}} ^ {*} (s ^ {\prime}) \right|} \\ & {\qquad \leq \gamma \sum_ {s ^ {\prime}} \overline {{p}} _ {t} (s ^ {\prime} \mid s, a) \left| V (\nu^ {s ^ {\prime}}) - \overline {{V}} _ {t _ {0}, t ^ {\prime}} ^ {*} (s ^ {\prime}) \right|} \\ & {\qquad \leq \gamma \sum_ {s ^ {\prime}} \overline {{p}} _ {t} (s ^ {\prime} \mid s, a) \gamma^ {(d _ {\max} - (d + 1))} \delta} \\ & {\qquad \leq \gamma^ {(d _ {\max} - d)} \delta} \end{array}
$$

Decision nodes case. Consider now any decision node $\nu ^ { s , t }$ at the same depth $d \in [ 0 , d _ { \mathrm { m a x } } )$ . The value of such a node is given by Equation 5 of the paper and the following holds.

$$
V (\nu^ {s, t}) = V (\nu^ {s, t, \bar {a}}), \text {with,} \bar {a} = \underset {a \in \mathcal {A}} {\arg \max} V (\nu^ {s, t, a})
$$

Similarly, we define $a ^ { * } \in { \mathcal { A } }$ as follows:

$$
\overline {{V}} _ {t _ {0}, t} ^ {*} (s) = \overline {{Q}} _ {t _ {0}, t} ^ {*} (s, a ^ {*}), \text {with,} a ^ {*} = \underset {a \in \mathcal {A}} {\arg \max} \overline {{Q}} _ {t _ {0}, t} ^ {*} (s, a)
$$

We distinguish two cases: 1) if $\bar { a } = a ^ { * }$ and 2) if $\bar { a } \neq a ^ { * }$ . In case 1), the result is trivial by writing the value of the decision node as the value of the chance node with the action $a ^ { * }$ and using the – already proven for depth d – result of Equation 10.

$$
\begin{array}{c} | V (\nu^ {s, t}) - \overline {{V}} _ {t _ {0}, t} ^ {*} (s) | = | V (\nu^ {s, t, a ^ {*}}) - \overline {{Q}} _ {t _ {0}, t} ^ {*} (s, a ^ {*}) | \\ \leq \gamma^ {(d _ {\max} - d)} \delta \end{array}
$$

In case $2 ) .$ , the maximizing actions are different. Still following Equation 10, we have that $V ( \nu ^ { s , t , a ^ { * } } ) \geq \overline { { Q } } _ { t _ { 0 } , t } ^ { * } ( s , a ^ { * } ) - \gamma ^ { ( d _ { \operatorname* { m a x } } - d ) } \delta$ . Yet, since a¯ is the maximizing action in the tree, we have that $V ( \nu ^ { s , t , \bar { a } } ) \geq V ( \nu ^ { s , t , a ^ { * } } )$ . By transitivity, we can thus write the following:

$$
\begin{array}{r l r} & & V (\nu^ {s, t, \bar {a}}) \geq \overline {{Q}} _ {t _ {0}, t} ^ {*} (s, a ^ {*}) - \gamma^ {(d _ {\max} - d)} \delta \\ & \Rightarrow & \overline {{Q}} _ {t _ {0}, t} ^ {*} (s, a ^ {*}) - V (\nu^ {s, t, \bar {a}}) \leq \gamma^ {(d _ {\max} - d)} \delta \end{array}\tag{11}
$$

Furthermore, still following Equation 10, we have that $\overline { { Q } } _ { t _ { 0 } , t } ^ { * } ( s , \bar { a } ) \geq V ( \nu ^ { s , t , \bar { a } } ) - \gamma ^ { ( d _ { \operatorname* { m a x } } - d ) } \delta$ . Yet, since $a ^ { * }$ is the maximizing action in MDP[ , we have that $\overline { { Q } } _ { t _ { 0 } , t } ^ { * } ( s , a ^ { * } ) \geq \overline { { Q } } _ { t _ { 0 } , t } ^ { * } ( s , \bar { a } )$ . By transitivity, we can thus write the following:

$$
\begin{array}{r l r} & & {\overline {{Q}} _ {t _ {0}, t} ^ {*} (s, a ^ {*}) \geq V (\nu^ {s, t, \bar {a}}) - \gamma^ {(d _ {\max} - d)} \delta} \\ & \Rightarrow & {V (\nu^ {s, t, \bar {a}}) - \overline {{Q}} _ {t _ {0}, t} ^ {*} (s, a ^ {*}) \leq \gamma^ {(d _ {\max} - d)} \delta} \end{array}\tag{12}
$$

By assembling equations 11 and 12, we prove equation 9 and the proof by induction is complete.

## 6 Proof of Property 6

Let $s , t _ { 0 } , t \in \mathcal { S } \times \mathcal { T } \times \mathcal { T }$ be. We consider the two snapshots $\mathrm { M D P } _ { t _ { 0 } }$ and $\mathrm { \mathbf { M D P } } _ { t }$ and are interested in the values of s within those two snapshots using the random policy π. We note $V _ { \mathrm { M D P } _ { t _ { 0 } } } ^ { \pi } \left( s \right)$ and $V _ { \mathrm { M D P } _ { t } } ^ { \pi } \left( s \right)$ those values. Let $n \in \mathbb N$ be. We note $V _ { \mathrm { M D P } _ { t _ { 0 } } } ^ { \pi , n } \left( s \right)$ and $V _ { \mathrm { M D P } _ { t } } ^ { \pi , n } \left( s \right)$ the finite horizon values defined as follows:

$$
V _ {\mathrm{MDP} _ {t _ {0}}} ^ {\pi , n} (s) = \mathbb {E} \left\{\sum_ {i = 0} ^ {n} \gamma^ {i} r _ {t _ {0}} (s _ {i}, a _ {i}, s _ {i + i}) \Bigg | \begin{array}{l} s _ {0} = s, \\ s _ {i + 1} \sim p _ {t _ {0}} (\cdot \mid s _ {i}, a _ {i}), i \geq 0 \\ a _ {i} \sim \pi (\cdot), i \geq 0 \end{array} \right\}
$$

where we replace $t _ { 0 }$ by t for the definition of $V _ { \mathrm { M D P } _ { t } } ^ { \pi , n } \left( s \right)$ . We first prove a result on the finite horizon values in Lemma 2.

Lemma 2. We consider an $( L _ { p } , L _ { R } ) – L C – N S M D P .$ . For $s , t , t _ { 0 } \in \mathcal { S } \times \mathcal { T } \times \mathcal { T }$ and $n \in \mathbb { N } ,$ , the finite horizon of the values of s within the snapshots $M D P _ { t }$ and $M D P _ { t _ { 0 } }$ verify:

$$
| V _ {M D P _ {t _ {0}}} ^ {\pi , n} (s) - V _ {M D P _ {t}} ^ {\pi , n} (s) | \leq L _ {V _ {n}} | t - t _ {0} |
$$

$$
w i t h, L _ {V _ {n}} = \sum_ {i = 0} ^ {n} \gamma^ {i} L _ {R}
$$

Proof. The proof is made by induction. Let us start with $n = 0$ . By definition, we have:

$$
\begin{array}{r l} & {\left| V _ {\mathrm{MDP} _ {t _ {0}}} ^ {\pi , 0} (s) - V _ {\mathrm{MDP} _ {t}} ^ {\pi , 0} (s) \right| = \left| \int_ {\mathcal {A}} \pi (a \mid s) \left(R _ {t _ {0}} (s, a) - R _ {t} (s, a)\right) d a \right|} \\ & {\qquad \leq \int_ {\mathcal {A}} \pi (a \mid s) L _ {R} | t _ {0} - t | d a} \\ & {\qquad \leq L _ {R} | t _ {0} - t |} \end{array}
$$

Which verifies the property for $n = 0$ with $L _ { V _ { 0 } } = L _ { R }$ . Let us now consider $n \in \mathbb { N }$ and suppose the property true for rank $n - 1$ . By writing the Bellman equation for the two value functions, we obtain the following calculation:

$$
\begin{array}{c} V _ {\mathrm{MDP} _ {t _ {0}}} ^ {\pi , n} (s) - V _ {\mathrm{MDP} _ {t}} ^ {\pi , n} (s) = \int_ {\mathcal {S} \times \mathcal {A}} \pi (a | s) \Big [ p _ {t _ {0}} (s ^ {\prime} \mid s, a) (r _ {t _ {0}} (s, a, s ^ {\prime}) + \gamma V _ {\mathrm{MDP} _ {t _ {0}}} ^ {\pi , n - 1} (s ^ {\prime})) - \\ p _ {t} (s ^ {\prime} \mid s, a) (r _ {t} (s, a, s ^ {\prime}) + \gamma V _ {\mathrm{MDP} _ {t}} ^ {\pi , n - 1} (s ^ {\prime})) \Big ] d s ^ {\prime} d a \end{array}
$$

$$
\text {i.e.} V _ {\mathrm{MDP} _ {t _ {0}}} ^ {\pi , n} (s) - V _ {\mathrm{MDP} _ {t}} ^ {\pi , n} (s) = \int_ {\mathcal {A}} \pi (a | s) \Big [ A (s, a) + B (s, a) \Big ] d a\tag{13}
$$

With the following values for $A ( s , a )$ and $B ( s , a )$

$$
\begin{array}{l} A (s, a) = \int_ {\mathcal {S}} (r _ {t _ {0}} (s, a, s ^ {\prime}) + \gamma V _ {\mathrm{MDP} _ {t _ {0}}} ^ {\pi , n - 1} (s ^ {\prime})) \Big [ p _ {t _ {0}} (s ^ {\prime} \mid s, a) - p _ {t} (s ^ {\prime} \mid s, a) \Big ] d s ^ {\prime} \\ B (s, a) = \int_ {\mathcal {S}} p _ {t} (s ^ {\prime} \mid s, a) \Big [ r _ {t _ {0}} (s, a, s ^ {\prime}) - r _ {t} (s, a, s ^ {\prime}) + \gamma (V _ {\mathrm{MDP} _ {t _ {0}}} ^ {\pi , n - 1} (s ^ {\prime}) - V _ {\mathrm{MDP} _ {t}} ^ {\pi , n - 1} (s ^ {\prime})) \Big ] d s ^ {\prime} \end{array}
$$

Let us first bound $A ( s , a )$ by noticing that $s ^ { \prime } \mapsto r _ { t _ { 0 } } ( s , a , s ^ { \prime } ) + \gamma V _ { \mathrm { M D P } _ { t _ { 0 } } } ^ { \pi , n - 1 } ( s ^ { \prime } )$ is bounded by $\frac { 1 } { 1 - \gamma }$ Since the function $\begin{array} { r } { s ^ { \prime } \mapsto \frac { 1 } { 1 - \gamma } } \end{array}$ belongs to $\mathrm { L i p } _ { 1 }$ , we can write the following:

$$
\begin{array}{l} A (s, a) \leq \sup _ {f \in \mathrm{Lip} _ {1}} \int_ {\mathcal {S}} f (s ^ {\prime}) \Big [ p _ {t _ {0}} (s ^ {\prime} \mid s, a) - p _ {t} (s ^ {\prime} \mid s, a) \Big ] d s ^ {\prime} \\ \qquad \leq W _ {1} (p _ {t _ {0}}, p _ {t}) \\ \qquad \leq L _ {p} | t - t _ {0} | \end{array}
$$

B is straightforwardly bounded using the induction hypothesis:

$$
\begin{array}{c} B (s, a) \leq \int_ {\mathcal {S}} p _ {t} (s ^ {\prime} \mid s, a) \Big [ L _ {r} | t - t _ {0} | + \gamma \sum_ {i = 0} ^ {n - 1} \gamma^ {i} L _ {R} | t - t _ {0} | \Big ] d s ^ {\prime} \\ \leq L _ {r} | t - t _ {0} | + \sum_ {i = 1} ^ {n} \gamma^ {i} L _ {R} | t - t _ {0} | \end{array}
$$

We inject the result in Equation 13:

$$
\begin{array}{l} V _ {\mathrm{MDP} _ {t _ {0}}} ^ {\pi , n} (s) - V _ {\mathrm{MDP} _ {t}} ^ {\pi , n} (s) \leq \int_ {\mathcal {A}} \pi (a | s) \Big [ L _ {p} | t - t _ {0} | + L _ {r} | t - t _ {0} | + \sum_ {i = 1} ^ {n} \gamma^ {i} L _ {R} | t - t _ {0} | \Big ] d a \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \\ \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquada \\ \qquad \qquad \qquad \leq (L _ {p} + L _ {r}) | t - t _ {0} | + \sum_ {i = 1} ^ {n} \gamma^ {i} L _ {R} | t - t _ {0} | \\ \qquad \qquad \qquad \leq L _ {R} | t - t _ {0} | + \sum_ {i = 1} ^ {n} \gamma^ {i} L _ {R} | t - t _ {0} | \\ \qquad \qquad \qquad \leq \sum_ {i = 0} ^ {n} \gamma^ {i} L _ {R} | t - t _ {0} | \\ \end{array}
$$

The same result can be derived with the opposite expression. Hence, taking the absolute value, we prove the property at rank n, i.e.

$$
| V _ {\mathrm{MDP} _ {t _ {0}}} ^ {\pi , n} (s) - V _ {\mathrm{MDP} _ {t}} ^ {\pi , n} (s) | \leq \sum_ {i = 0} ^ {n} \gamma^ {i} L _ {R} | t - t _ {0} |\tag{14}
$$

which concludes the proof by induction.

The proof of Property 6 follows easily by remarking that the sequence $L _ { V _ { n } }$ of Lemma 2 is geometric and converges towards $\frac { L _ { R } } { 1 - \gamma }$ when n goes to infinity.

![](images/602f3dfdb459178686181ea30b7095abbe35c9d69f681a0b047e6ed81aab4c4b.jpg)  
Figure 1: The Non-Stationary bridge environment

## 7 Non-Stationary bridge environment

## 8 Informations about the Machine Learning reproducibility checklist

For the experiments run in Section 6, the computing infrastructure used was a laptop using four 64-bit CPU (model: Intel(R) Core(TM) i7-4810MQ CPU @ 2.80GHz). The collected samples sizes and number of evaluation runs for each experiment are summarized in Table 1.

<table><tr><td>Experiment</td><td>Number of experiment repetitions</td><td>Number of episodes</td><td>Maximum length of episodes</td><td>Upper bound on the number of computed transition samples (s, a, r, s&#x27;)</td></tr><tr><td>Non-Stationary BridgeFigure 1</td><td>3(one per agent)</td><td>96</td><td>10</td><td>89,579,520</td></tr></table>

Table 1: Summary of the number of experiment repetition, number of sampled tasks, number of episodes, maximum length of episodes and upper bounds on the number of collected samples.

The displayed confidence intervals in Figure 2a is 50% of the estimated confidence interval σ¯ computed w.r.t. the following formula:

$$
\bar {\sigma} = \sqrt {\frac {1}{1 - N} \sum_ {i = 1} ^ {N} (x _ {i} - \bar {x}) ^ {2}} \quad \text {where,} \quad \bar {x} = \frac {1}{N} \sum_ {i = 1} ^ {N} x _ {i},
$$

with $D = \{ x _ { i } \} _ { i = 1 } ^ { N }$ the set of the collected data (discounted return in this case). No data were excluded neither pre-computed. Hyper-parameters were determined to our appreciation, they may be sub-optimal but we found the results convincing enough to display interesting behaviours.

## References

Pravin M. Vaidya. Speeding-up linear programming using fast matrix multiplication. In 30th Annual Symposium on Foundations of Computer Science, pages 332–337. IEEE, 1989.