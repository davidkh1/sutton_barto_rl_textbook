"""
Figure 2.4: Average performance of UCB action selection on the 10-armed testbed.

Compares UCB (c=2) with ε-greedy (ε=0.1), both using sample averages.
UCB generally performs better, except in the first k steps when it selects
randomly among untried actions.

UCB action selection (equation 2.10):
    A_t = argmax_a [Q_t(a) + c * sqrt(ln(t) / N_t(a))]

From the RL book:
"Figure 2.4: Average performance of UCB action selection on the 10-armed testbed.
As shown, UCB generally performs better than ε-greedy action selection, except in
the first k steps, when it selects randomly among the as-yet-untried actions."
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from multi_armed_testbed import MultiArmedTestbed

OUTPUT_FILE = Path(__file__).parent / 'output' / 'figure_2_4.png'

N_RUNS = 2000
N_STEPS = 1000


def run_ucb(q_star: np.ndarray, n_steps: int, c: float, seed: int = 42,
            ) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized UCB simulation across all runs.

    UCB action selection (equation 2.10):
        A_t = argmax_a [Q_t(a) + c * sqrt(ln(t) / N_t(a))]
    Untried actions (N_t(a) = 0) are selected first.
    """
    n_runs, n_arms = q_star.shape
    rng = np.random.default_rng(seed)

    q_estimates = np.zeros((n_runs, n_arms))
    action_counts = np.zeros((n_runs, n_arms), dtype=int)
    avg_rewards = np.zeros(n_steps)
    pct_optimal = np.zeros(n_steps)
    optimal_actions = np.argmax(q_star, axis=1)
    run_idx = np.arange(n_runs)

    for t in range(n_steps):
        # UCB bonus: c * sqrt(ln(t+1) / N_t(a)), with infinity for untried actions
        safe_counts = np.maximum(action_counts, 1)
        ucb_bonus = c * np.sqrt(np.log(t + 1) / safe_counts)
        ucb_bonus[action_counts == 0] = np.inf

        # Tiebreak randomly
        ucb_values = q_estimates + ucb_bonus + rng.random((n_runs, n_arms)) * 1e-10
        actions = np.argmax(ucb_values, axis=1)

        # Get rewards
        rewards = rng.normal(q_star[run_idx, actions], 1.0)

        # Update Q estimates (sample average)
        action_counts[run_idx, actions] += 1
        step_sizes = 1.0 / action_counts[run_idx, actions]
        q_estimates[run_idx, actions] += step_sizes * (rewards - q_estimates[run_idx, actions])

        # Record results
        avg_rewards[t] = rewards.mean()
        pct_optimal[t] = (actions == optimal_actions).mean() * 100

    return avg_rewards, pct_optimal


def run_eps_greedy(q_star: np.ndarray, n_steps: int, epsilon: float, seed: int = 42,
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized ε-greedy simulation (sample averages)."""
    n_runs, n_arms = q_star.shape
    rng = np.random.default_rng(seed)

    q_estimates = np.zeros((n_runs, n_arms))
    action_counts = np.zeros((n_runs, n_arms), dtype=int)
    avg_rewards = np.zeros(n_steps)
    pct_optimal = np.zeros(n_steps)
    optimal_actions = np.argmax(q_star, axis=1)
    run_idx = np.arange(n_runs)

    for t in range(n_steps):
        explore = rng.random(n_runs) < epsilon
        random_actions = rng.integers(n_arms, size=n_runs)
        noisy_q = q_estimates + rng.random((n_runs, n_arms)) * 1e-10
        greedy_actions = np.argmax(noisy_q, axis=1)
        actions = np.where(explore, random_actions, greedy_actions)

        rewards = rng.normal(q_star[run_idx, actions], 1.0)

        action_counts[run_idx, actions] += 1
        step_sizes = 1.0 / action_counts[run_idx, actions]
        q_estimates[run_idx, actions] += step_sizes * (rewards - q_estimates[run_idx, actions])

        avg_rewards[t] = rewards.mean()
        pct_optimal[t] = (actions == optimal_actions).mean() * 100

    return avg_rewards, pct_optimal


def plot_figure_2_4(results: dict[str, tuple]) -> None:
    """Plot average reward over time for UCB vs ε-greedy."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {'UCB c=2': 'blue', 'ε-greedy ε=0.1': 'gray'}
    for name, (avg_rewards, _) in results.items():
        ax.plot(avg_rewards, label=name, color=colors[name], linewidth=1.2)

    ax.set_xlabel('Steps')
    ax.set_ylabel('Average\nreward', rotation=0, labelpad=40, va='center')
    ax.set_xlim(0, N_STEPS)
    ax.set_ylim(0, 1.6)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Figure 2.4: UCB action selection on the 10-armed testbed')

    plt.tight_layout()
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=150)
    plt.show()


if __name__ == '__main__':
    testbed = MultiArmedTestbed(n_arms=10, n_tasks=N_RUNS, seed=42)

    print("Running UCB c=2...")
    ucb_results = run_ucb(testbed.q_star, N_STEPS, c=2.0, seed=43)

    print("Running ε-greedy ε=0.1...")
    eps_results = run_eps_greedy(testbed.q_star, N_STEPS, epsilon=0.1, seed=43)

    experiment_results = {
        'UCB c=2': ucb_results,
        'ε-greedy ε=0.1': eps_results,
    }

    plot_figure_2_4(experiment_results)
    print(f"Saved {OUTPUT_FILE}")
