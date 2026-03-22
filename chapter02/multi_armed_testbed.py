"""
Multi-Armed Testbed (Section 2.3)

Reusable k-armed bandit testbed for comparing action-value methods.
- 2000 randomly generated k-armed bandit problems (default k=10)
- True action values q*(a) sampled from N(0, 1)
- Actual rewards sampled from N(q*(a), 1)

Reference in Lisp: http://incompleteideas.net/book/code/testbed.lisp

Description from the RL book:
  2.3 The 10-armed Testbed
To roughly assess the relative effectiveness of the greedy and eps-greedy action-value
methods, we compared them numerically on a suite of test problems. This was a set
of 2000 randomly generated k-armed bandit problems with k = 10. For each bandit
problem, such as the one shown in Figure 2.1, the action values, q*(a), a = 1, ..., 10,
were selected according to a normal (Gaussian) distribution with mean 0 and variance 1.
Then, when a learning method applied to that problem selected action A(t) at time step t,
the actual reward, R(t), was selected from a normal distribution with mean q*(A(t)) and
variance 1. These distributions are shown in gray in Figure 2.1. We call this suite of test
tasks the 10-armed testbed. For any learning method, we can measure its performance
and behavior as it improves with experience over 1000 time steps when applied to one of
the bandit problems. This makes up one run. Repeating this for 2000 independent runs,
each with a different bandit problem, we obtained measures of the learning algorithm’s
average behavior.
"""

import numpy as np


class MultiArmedTestbed:
    """k-armed bandit testbed for comparing action-value methods."""

    def __init__(self, n_arms: int = 10, n_tasks: int = 2000, seed: int = 42):
        self.n_arms = n_arms
        self.n_tasks = n_tasks
        self.seed = seed  # consistency in results
        self.rng = np.random.default_rng(seed)

        # True action values q*(a) for each task, sampled from N(0, 1)
        self.q_star = self.rng.standard_normal((n_tasks, n_arms))

        # Optimal action for each task
        self.optimal_actions = np.argmax(self.q_star, axis=1)  # (n_tasks,) 1D array

        # Per-run state (initialized by reset())
        self.Q = None
        self.action_counts = None
        self.current_task = None

    def reset_rng(self) -> None:
        """Reset RNG to initial state for reproducible comparisons."""
        self.rng = np.random.default_rng(self.seed)

    def reset(self, task: int) -> None:
        """Reset Q estimates and action counts for a new run."""
        self.Q = np.zeros(self.n_arms)  # Action-value estimates
        self.action_counts = np.zeros(self.n_arms, dtype=int)  # Action counts
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
            return int(self.rng.integers(self.n_arms))
        else:
            # Random tiebreak among actions with max Q
            max_q = np.max(self.Q)
            max_actions = np.where(self.Q == max_q)[0]
            return self.rng.choice(max_actions)

    def learn(self, action: int, reward: float) -> None:
        """Update action-value estimate using sample average."""
        self.action_counts[action] += 1
        # Incremental update: Q += (r - Q) / n
        self.Q[action] += (reward - self.Q[action]) / self.action_counts[action]

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


def run_vectorized(
    q_star: np.ndarray,
    n_steps: int,
    epsilon: float,
    alpha: float | None = None,
    q_init: float = 0.0,
    walk_std: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized bandit simulation across all runs in parallel.

    Args:
        q_star: True action values, shape (n_runs, n_arms). Copied internally;
                the original is not modified.
        n_steps: Number of time steps per run.
        epsilon: Exploration probability for epsilon-greedy.
        alpha: Constant step-size. None for sample-average (1/n).
        q_init: Initial Q estimate for all actions.
        walk_std: Std of random walk added to q* each step (0 = stationary).
        seed: RNG seed for action selection and rewards.

    Returns:
        avg_rewards: Average reward at each time step, shape (n_steps,).
        pct_optimal: % optimal action at each time step, shape (n_steps,).
    """
    q_star = q_star.copy()
    n_runs, n_arms = q_star.shape
    rng = np.random.default_rng(seed)

    q_estimates = np.full((n_runs, n_arms), q_init)
    action_counts = np.zeros((n_runs, n_arms), dtype=int)
    avg_rewards = np.zeros(n_steps)
    pct_optimal = np.zeros(n_steps)
    run_idx = np.arange(n_runs)

    for t in range(n_steps):
        # Epsilon-greedy action selection
        explore = rng.random(n_runs) < epsilon
        random_actions = rng.integers(n_arms, size=n_runs)
        noisy_q = q_estimates + rng.random((n_runs, n_arms)) * 1e-10
        greedy_actions = np.argmax(noisy_q, axis=1)
        actions = np.where(explore, random_actions, greedy_actions)

        # Get rewards: R ~ N(q*(action), 1)
        rewards = rng.normal(q_star[run_idx, actions], 1.0)

        # Update Q estimates
        errors = rewards - q_estimates[run_idx, actions]
        if alpha is None:
            action_counts[run_idx, actions] += 1
            step_sizes = 1.0 / action_counts[run_idx, actions]
        else:
            step_sizes = alpha
        q_estimates[run_idx, actions] += step_sizes * errors

        # Record results
        avg_rewards[t] = rewards.mean()
        optimal_actions = np.argmax(q_star, axis=1)
        pct_optimal[t] = (actions == optimal_actions).mean() * 100

        # Random walk (nonstationary)
        if walk_std > 0:
            q_star += rng.normal(0, walk_std, (n_runs, n_arms))

    return avg_rewards, pct_optimal
