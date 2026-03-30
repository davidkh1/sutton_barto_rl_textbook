"""
Figure 2.5: Gradient bandit algorithm with and without reward baseline.

Compares four configurations on a variant of the 10-armed testbed where
q*(a) ~ N(+4, 1) instead of N(0, 1):
  - α = 0.1, with baseline
  - α = 0.4, with baseline
  - α = 0.1, without baseline
  - α = 0.4, without baseline

The +4 offset has no effect when the baseline is used (it adapts), but
significantly degrades performance without a baseline.

From the RL book:
"Figure 2.5: Average performance of the gradient bandit algorithm with and without
a reward baseline on the 10-armed testbed when the q*(a) are chosen to be near +4
rather than near zero."
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_FILE = Path(__file__).parent / 'output' / 'figure_2_5.png'

N_RUNS = 2000
N_STEPS = 1000
N_ARMS = 10
Q_STAR_MEAN = 4.0


def run_gradient_bandit(
    q_star: np.ndarray,
    n_steps: int,
    alpha: float,
    use_baseline: bool,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized gradient bandit simulation.

    Uses soft-max action probabilities (equation 2.11) and preference
    updates (equation 2.12).
    """
    n_runs, n_arms = q_star.shape
    rng = np.random.default_rng(seed)

    H = np.zeros((n_runs, n_arms))          # preferences
    avg_reward = np.zeros(n_runs)            # running average (baseline)
    avg_rewards = np.zeros(n_steps)
    pct_optimal = np.zeros(n_steps)
    optimal_actions = np.argmax(q_star, axis=1)
    run_idx = np.arange(n_runs)

    for t in range(n_steps):
        # Soft-max probabilities (equation 2.11)
        # Subtract max for numerical stability
        H_stable = H - H.max(axis=1, keepdims=True)
        exp_H = np.exp(H_stable)
        pi = exp_H / exp_H.sum(axis=1, keepdims=True)

        # Sample actions from the probability distribution
        cumulative = pi.cumsum(axis=1)
        random_vals = rng.random((n_runs, 1))
        actions = (random_vals >= cumulative).sum(axis=1)
        actions = np.clip(actions, 0, n_arms - 1)

        # Get rewards
        rewards = rng.normal(q_star[run_idx, actions], 1.0)

        # Baseline
        baseline = avg_reward if use_baseline else 0.0

        # Update preferences (equation 2.12)
        one_hot = np.zeros((n_runs, n_arms))
        one_hot[run_idx, actions] = 1.0
        H += alpha * (rewards[:, np.newaxis] - baseline[:, np.newaxis] if use_baseline
                      else rewards[:, np.newaxis] - 0.0) * (one_hot - pi)

        # Update running average reward
        avg_reward += (rewards - avg_reward) / (t + 1)

        # Record results
        avg_rewards[t] = rewards.mean()
        pct_optimal[t] = (actions == optimal_actions).mean() * 100

    return avg_rewards, pct_optimal


def plot_figure_2_5(results: dict[str, tuple]) -> None:
    """Plot % optimal action for gradient bandit variants."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {
        'α=0.1, with baseline': '#cc8800',
        'α=0.4, with baseline': '#888888',
        'α=0.1, without baseline': '#cc8800',
        'α=0.4, without baseline': '#888888',
    }
    linestyles = {
        'α=0.1, with baseline': '-',
        'α=0.4, with baseline': '-',
        'α=0.1, without baseline': '--',
        'α=0.4, without baseline': '--',
    }

    for name, (_, pct_optimal) in results.items():
        ax.plot(pct_optimal, label=name, color=colors[name],
                linestyle=linestyles[name], linewidth=1.2)

    ax.set_xlabel('Steps')
    ax.set_ylabel('% Optimal\naction', rotation=0, labelpad=40, va='center')
    ax.set_xlim(0, N_STEPS)
    ax.set_ylim(0, 100)
    ax.legend(loc='center right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Figure 2.5: Gradient bandit on the 10-armed testbed '
                 f'(q* near +{Q_STAR_MEAN:.0f})')

    plt.tight_layout()
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=150)
    plt.show()


if __name__ == '__main__':
    rng = np.random.default_rng(42)
    q_star = rng.normal(Q_STAR_MEAN, 1.0, (N_RUNS, N_ARMS))

    configs = [
        ('α=0.1, with baseline',    0.1, True),
        ('α=0.4, with baseline',    0.4, True),
        ('α=0.1, without baseline', 0.1, False),
        ('α=0.4, without baseline', 0.4, False),
    ]

    results = {}
    for name, alpha, use_baseline in configs:
        print(f"Running {name}...")
        results[name] = run_gradient_bandit(
            q_star, N_STEPS, alpha, use_baseline, seed=43,
        )

    plot_figure_2_5(results)
    print(f"Saved {OUTPUT_FILE}")
