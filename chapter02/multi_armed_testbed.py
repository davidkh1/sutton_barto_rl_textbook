"""
Multi-Armed Testbed (Section 2.3)

Reusable k-armed bandit testbed for comparing action-value methods.
- 2000 randomly generated k-armed bandit problems (default k=10)
- True action values q*(a) sampled from N(0, 1)
- Actual rewards sampled from N(q*(a), 1)

Reference: http://incompleteideas.net/book/code/testbed.lisp
"""

import numpy as np


class MultiArmedTestbed:
    """k-armed bandit testbed for comparing action-value methods."""

    def __init__(self, n_arms: int = 10, n_tasks: int = 2000, seed: int = 42):
        self.n_arms = n_arms
        self.n_tasks = n_tasks
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # True action values q*(a) for each task, sampled from N(0, 1)
        self.q_star = self.rng.standard_normal((n_tasks, n_arms))

        # Optimal action for each task
        self.optimal_actions = np.argmax(self.q_star, axis=1)

    def reset_rng(self) -> None:
        """Reset RNG to initial state for reproducible comparisons."""
        self.rng = np.random.default_rng(self.seed)

    def reset(self, task: int) -> None:
        """Reset Q estimates and action counts for a new run."""
        self.Q = np.zeros(self.n_arms)  # Action-value estimates
        self.n_a = np.zeros(self.n_arms, dtype=int)  # Action counts
        self.current_task = task

    def reward(self, action: int) -> float:
        """Sample reward from N(q*(a), 1)."""
        return self.rng.normal(self.q_star[self.current_task, action], 1.0)

    def optimal_action(self) -> int:
        """Return optimal action for current task."""
        return self.optimal_actions[self.current_task]

    def epsilon_greedy(self, epsilon: float) -> int:
        """Select action using epsilon-greedy policy."""
        if self.rng.random() < epsilon:
            return self.rng.integers(self.n_arms)
        else:
            # Random tiebreak among actions with max Q
            max_q = np.max(self.Q)
            max_actions = np.where(self.Q == max_q)[0]
            return self.rng.choice(max_actions)

    def learn(self, action: int, reward: float) -> None:
        """Update action-value estimate using sample average."""
        self.n_a[action] += 1
        # Incremental update: Q += (r - Q) / n
        self.Q[action] += (reward - self.Q[action]) / self.n_a[action]

    def run(self, n_steps: int, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Execute one run of the bandit problem.

        Returns:
            rewards: Array of rewards at each time step
            optimal: Array of 1s where optimal action was selected, 0s otherwise
        """
        rewards = np.zeros(n_steps)
        optimal = np.zeros(n_steps)
        opt_action = self.optimal_action()

        for t in range(n_steps):
            action = self.epsilon_greedy(epsilon)
            r = self.reward(action)
            self.learn(action, r)

            rewards[t] = r
            optimal[t] = 1 if action == opt_action else 0

        return rewards, optimal

    def runs(self, n_runs: int = 2000, n_steps: int = 1000, epsilon: float = 0.0
             ) -> tuple[np.ndarray, np.ndarray]:
        """
        Execute multiple runs and compute average performance.

        Returns:
            avg_rewards: Average reward at each time step
            pct_optimal: Percentage of optimal action selections at each time step
        """
        all_rewards = np.zeros((n_runs, n_steps))
        all_optimal = np.zeros((n_runs, n_steps))

        for run in range(n_runs):
            self.reset(task=run)
            rewards, optimal = self.run(n_steps, epsilon)
            all_rewards[run] = rewards
            all_optimal[run] = optimal

        return all_rewards.mean(axis=0), all_optimal.mean(axis=0) * 100
