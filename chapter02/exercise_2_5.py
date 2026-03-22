"""
Exercise 2.5: Nonstationary Bandit Problem (Section 2.5)

Design and conduct an experiment to demonstrate the difficulties that sample-average
methods have for nonstationary problems. Use a modified version of the 10-armed testbed
in which all the q*(a) start out equal and then take independent random walks (say by
adding a normally distributed increment with mean 0 and standard deviation 0.01 to all
the q*(a) on each step). Prepare plots like Figure 2.2 for an action-value method using
sample averages, incrementally computed, and another action-value method using a
constant step-size parameter, alpha = 0.1. Use epsilon = 0.1 and longer runs, say of
10,000 steps.

Analysis of results
-------------------
The average reward curve looks very different from Figure 2.2:

1. Much lower rewards in the first 1000 steps. In Figure 2.2, q*(a) are sampled from
   N(0,1), so the best arm starts around 1.54. Here all q*(a) start at 0 and only
   diverge slowly via random walks (std=0.01 per step), so there is very little
   difference between arms early on — the best arm's advantage is tiny.

2. Rewards keep climbing instead of leveling off. Each q*(a) follows a random walk,
   so Var(q*(a)) = t * 0.01^2 grows linearly with time. The spread between arms
   increases as sqrt(t), meaning the best arm's value drifts higher and higher.
   Unlike Figure 2.2 where the best possible reward is fixed at ~1.54, here there
   is no ceiling — the expected best-arm value grows as ~0.01 * sqrt(t) * sqrt(2*ln(k)).
   In theory it's unbounded, but very slow: reaching average reward of 10 would
   require on the order of millions of steps.

3. The constant step-size (alpha=0.1) consistently outperforms sample averaging.
   Sample averaging weights all past rewards equally, so it cannot adapt when q*
   values drift. The exponential recency-weighted average forgets old observations
   and tracks the changing values, which is exactly the point of Exercise 2.5.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_FILE = Path(__file__).parent / 'output' / 'exercise_2_5.png'


@dataclass
class ExperimentConfig:
    n_arms: int = 10
    n_runs: int = 2000
    n_steps: int = 10000
    epsilon: float = 0.1
    walk_std: float = 0.01
    seed: int = 42


def run_nonstationary_experiment(
    cfg: ExperimentConfig = ExperimentConfig(),
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Run nonstationary bandit experiment comparing sample-average vs constant step-size.

    All q*(a) start at 0 and take independent random walks each step.

    Returns:
        dict mapping method name to (avg_rewards, pct_optimal) arrays.
    """
    methods = {
        'Sample average': None,       # step_size = 1/n
        'Constant α = 0.1': 0.1,      # step_size = 0.1
    }

    n_arms, n_runs, n_steps = cfg.n_arms, cfg.n_runs, cfg.n_steps
    epsilon, walk_std = cfg.epsilon, cfg.walk_std

    results = {}
    for name, alpha in methods.items():
        print(f"Running {name}...")
        rng = np.random.default_rng(cfg.seed)

        # Vectorize across all runs: shape (n_runs, n_arms)
        q_star = np.zeros((n_runs, n_arms))
        q_estimates = np.zeros((n_runs, n_arms))
        action_counts = np.zeros((n_runs, n_arms), dtype=int)

        avg_rewards = np.zeros(n_steps)
        avg_optimal = np.zeros(n_steps)

        for t in range(n_steps):
            # Epsilon-greedy action selection (vectorized across runs)
            explore = rng.random(n_runs) < epsilon
            random_actions = rng.integers(n_arms, size=n_runs)

            # Greedy actions: argmax with random tiebreaking
            # Add small noise to break ties randomly
            noisy_q = q_estimates + rng.random((n_runs, n_arms)) * 1e-10
            greedy_actions = np.argmax(noisy_q, axis=1)

            actions = np.where(explore, random_actions, greedy_actions)

            # Get rewards: R ~ N(q*(action), 1)
            q_selected = q_star[np.arange(n_runs), actions]
            rewards = rng.normal(q_selected, 1.0)

            # Update Q estimates
            action_mask = np.zeros((n_runs, n_arms), dtype=bool)
            action_mask[np.arange(n_runs), actions] = True
            action_counts += action_mask

            errors = np.zeros((n_runs, n_arms))
            errors[np.arange(n_runs), actions] = rewards - q_estimates[np.arange(n_runs), actions]

            if alpha is None:
                # Sample average: step_size = 1/n (avoid division by zero)
                safe_counts = np.maximum(action_counts, 1)
                step_sizes = np.where(action_counts > 0, 1.0 / safe_counts, 0.0)
                q_estimates += step_sizes * errors
            else:
                q_estimates += alpha * errors

            # Record results
            avg_rewards[t] = rewards.mean()
            optimal_actions = np.argmax(q_star, axis=1)
            avg_optimal[t] = (actions == optimal_actions).mean() * 100

            # Random walk: q*(a) += N(0, walk_std) for all actions and runs
            q_star += rng.normal(0, walk_std, (n_runs, n_arms))

        results[name] = (avg_rewards, avg_optimal)

    return results


def plot_exercise_2_5(results: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """Plot average reward and % optimal action, like Figure 2.2."""
    n_steps = len(next(iter(results.values()))[0])
    colors = {'Sample average': 'red', 'Constant α = 0.1': 'blue'}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Top plot: Average reward
    for name, (avg_rewards, _) in results.items():
        ax1.plot(avg_rewards, label=name, color=colors[name], linewidth=0.8)

    ax1.axvspan(0, 1000, alpha=0.08, color='orange')
    ax1.text(500, ax1.get_ylim()[1] * 0.95, 'Fig 2.2\nrange', ha='center', va='top',
             fontsize=9, color='#996600')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average\nreward', rotation=0, labelpad=40, va='center')
    ax1.set_xlim(0, n_steps)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Exercise 2.5: Nonstationary 10-armed testbed (ε = 0.1)')

    # Bottom plot: % Optimal action
    for name, (_, pct_optimal) in results.items():
        ax2.plot(pct_optimal, label=name, color=colors[name], linewidth=0.8)

    ax2.axvspan(0, 1000, alpha=0.08, color='orange')
    ax2.text(500, 95, 'Fig 2.2\nrange', ha='center', va='top',
             fontsize=9, color='#996600')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('% Optimal\naction', rotation=0, labelpad=40, va='center')
    ax2.set_xlim(0, n_steps)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=150)
    plt.show()


if __name__ == '__main__':
    cfg = ExperimentConfig()
    print("Exercise 2.5: Nonstationary bandit problem")
    print(f"Running {cfg.n_runs} runs x {cfg.n_steps:,} steps...")
    experiment_results = run_nonstationary_experiment(cfg)
    plot_exercise_2_5(experiment_results)
    print(f"Saved {OUTPUT_FILE}")
