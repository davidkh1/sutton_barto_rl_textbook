"""
Check what is the best reward per step on the N-armed Testbed.
For 10-armed Testbed: 1.538...->1.54
"""
import numpy as np
num_arms = 10  # as in the book
num_random_bandit_problems = 2_000_000
testbed = np.random.randn(num_random_bandit_problems, num_arms)
best_reward_per_problem = testbed.max(axis=1)  # size of num_random_bandit_problems
reward_per_testbed = best_reward_per_problem.mean()
print(reward_per_testbed)