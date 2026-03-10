"""
Estimate the best expected reward per step on the 10-armed Testbed (Section 2.3).
This is the upper bound in Figure 2.2: if an agent knew the true q*(a) values
and always picked the best arm, its average reward would converge to E[max_a q*(a)].
For 10 arms with q*(a) ~ N(0,1), that value is ≈ 1.5388 (rounded to 1.54).
"""
import numpy as np

rng = np.random.default_rng(seed=42)
num_arms = 10  # as in the book
num_problems = 2_000_000
testbed = rng.standard_normal((num_problems, num_arms))
best_reward_per_problem = testbed.max(axis=1)
best_reward_per_step = best_reward_per_problem.mean()
print(f"Best expected reward per step (10-armed): {best_reward_per_step:.4f}")