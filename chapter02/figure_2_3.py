"""
Figure 2.3: The effect of optimistic initial action-value estimates on the 10-armed testbed.

Compares two methods, both using constant step-size alpha = 0.1:
  - Optimistic, greedy: Q1 = +5, epsilon = 0
  - Realistic, epsilon-greedy: Q1 = 0, epsilon = 0.1

The optimistic method explores early because all initial estimates are much higher than
the true values (~1.54 at best). Every action "disappoints", causing the agent to try
other actions. Eventually it converges and outperforms the realistic epsilon-greedy method.

From the RL book:
"Figure 2.3: The effect of optimistic initial action-value estimates on the 10-armed testbed.
Both methods used a constant step-size parameter, alpha = 0.1."
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from multi_armed_testbed import MultiArmedTestbed, run_vectorized

OUTPUT_FILE = Path(__file__).parent / 'output' / 'figure_2_3.png'

N_RUNS = 2000
N_STEPS = 1000
ALPHA = 0.1


def run_figure_2_3(
    n_runs: int = N_RUNS,
    n_steps: int = N_STEPS,
    alpha: float = ALPHA,
    seed: int = 42,
) -> dict[str, tuple]:
    """
    Run optimistic vs realistic comparison on the stationary 10-armed testbed.

    Returns:
        dict mapping method name to (avg_rewards, pct_optimal).
    """
    testbed = MultiArmedTestbed(n_arms=10, n_tasks=n_runs, seed=seed)

    methods = {
        'Optimistic, greedy\n$Q_1$=5, ε=0': (5.0, 0.0),
        'Realistic, ε-greedy\n$Q_1$=0, ε=0.1': (0.0, 0.1),
    }

    results = {}
    for name, (q_init, epsilon) in methods.items():
        print(f"Running {name.replace(chr(10), ', ')}...")
        avg_rewards, pct_optimal = run_vectorized(
            testbed.q_star, n_steps, epsilon,
            alpha=alpha, q_init=q_init, seed=seed + 1,
        )
        results[name] = pct_optimal

    return results


def plot_figure_2_3(results: dict[str, np.ndarray]) -> None:
    """Plot % optimal action over time."""
    n_steps = len(next(iter(results.values())))

    fig, ax = plt.subplots(figsize=(10, 5))

    plot_colors = ['#4488cc', '#888888']
    for (name, pct_optimal), color in zip(results.items(), plot_colors):
        ax.plot(pct_optimal, label=name, color=color, linewidth=1.2)

    # Step 10 (0-indexed): the optimistic greedy agent has tried all 10 arms
    # (steps 0-9), each time being "disappointed" (reward << Q=5). On step 10
    # it picks the arm with the highest Q — likely the true best arm — causing
    # the first spike in % optimal action.
    ax.axvline(x=10, color='red', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(12, 48, 'step 10: all arms\ntried once', fontsize=8, color='red')

    ax.set_xlabel('Steps')
    ax.set_ylabel('% Optimal\naction', rotation=0, labelpad=40, va='center')
    ax.set_xlim(0, n_steps)
    ax.set_ylim(0, 100)
    ax.legend(loc='right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Figure 2.3: Optimistic initial values on the 10-armed testbed (α = 0.1)')

    plt.tight_layout()
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=150)
    plt.show()


if __name__ == '__main__':
    print("Generating Figure 2.3...")
    experiment_results = run_figure_2_3()
    plot_figure_2_3(experiment_results)
    print(f"Saved {OUTPUT_FILE}")
