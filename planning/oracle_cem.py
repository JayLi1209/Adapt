"""ORACLE CVaR-CEM: the SURF planner, but planning on the TRUE physics.

Identical to planning.cvar_cem.CVaRCEMAgent -- same I (CEM iterations), J
(candidates), K (models), N (rollouts per model), H (horizon), elite fraction,
CVaR alpha, planning discount, leaf value and policy-seeded proposal -- so it
spends exactly the same I*J*K*N*H simulated transitions per real step.  The
ONLY difference: the K transition models are not posterior draws from the BNN
but K copies of the exact transition kernel the environment is running.  There
is no learning, no forgetting and no epistemic uncertainty; the aleatoric slip
is still sampled in the N rollouts.

This is the "knows the physics" ceiling for the planner, as opposed to the
value-iteration oracles (run_frozenlake_oracle.py, Bridge/oracle_bridge.py),
which are the ceiling over ALL policies.
"""
import numpy as np

from planning.cvar_cem import CVaRCEMAgent


def true_kernel(grid, p):
    """Exact (S, A, S) kernel for intended-prob p under `grid`'s slip mode.

    For intended action a the K=3 outcome directions are grid.dir_actions(a)
    with probabilities grid.slip_dist(p); a move off the map self-transitions
    (grid.move clips), and goal/hole cells are absorbing -- exactly what
    env.gridworlds.build_env runs (same construction as Bridge/oracle_bridge.py).
    """
    S, A = grid.n_states, grid.n_actions
    T = np.zeros((S, A, S))
    dist = grid.slip_dist(p)
    flat = grid.flat_desc
    for s in range(S):
        if flat[s] in "GH":
            T[s, :, s] = 1.0
            continue
        for a in range(A):
            for k, d in enumerate(grid.dir_actions(a)):
                T[s, a, grid.move(s, d)] += dist[k]
    return T


class OracleCVaRCEMAgent(CVaRCEMAgent):
    """CVaR-CEM whose K 'posterior' models are all the true kernel."""

    def __init__(self, *args, true_T, **kwargs):
        super().__init__(*args, **kwargs)
        self.true_T = np.asarray(true_T, dtype=np.float64)

    def _model_matrices(self, k, deterministic):
        Ts = np.repeat(self.true_T[None], k, axis=0)
        # Rs (reward-head predictions) is never used for scoring by CVaR-CEM,
        # which scores with the map's reward-on-arrival vector.
        return Ts, np.zeros((k, self.n, self.n_actions))
