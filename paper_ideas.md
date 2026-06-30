## Pros
1. The rollout is purely imaginary (like Ada-MCTS)
2. No expensive Gradient Descent updates after change (just update the Direchlet *count* head, which reduces the compute)
3. It integrates uncertainty estimation into Mahalanobis distance calculation.
4. It has a risk knob (cVaR) that is better in controlling risk
5. It achieves similar/better accuracy than Ada-MCTS on a more difficult environment transition where the policy needs to change (optionally, bring out the Pareto Frontier idea)
6. One model (but still model-based), but avoids confidence but wrong predictions
7. It is versatile to all non-stationary envs and changes (reward changes, continuous envs, etc.)

## Cons
1. Requires a lot of hyperparameter tuning