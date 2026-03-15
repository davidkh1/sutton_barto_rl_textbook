"""
Figure 2.2: Average performance of epsilon-greedy action-value methods.

Compares greedy (ε=0), ε=0.01, and ε=0.1 on the 10-armed testbed.
Shows average reward and % optimal action over 1000 steps, averaged over 2000 runs.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from multi_armed_testbed import MultiArmedTestbed

OUTPUT_FILE = Path(__file__).parent / 'output' / 'figure_2_2.png'


def plot_figure_2_2(testbed: MultiArmedTestbed, n_runs: int = 2000, n_steps: int = 1000) -> None:
    epsilons = [0.0, 0.01, 0.1]
    labels = ['ε = 0 (greedy)', 'ε = 0.01', 'ε = 0.1']
    colors = ['green', 'red', 'blue']

    results = {}
    for eps in epsilons:
        print(f"Running epsilon = {eps}...")
        testbed.reset_rng()
        avg_reward, pct_optimal = testbed.runs(n_runs, n_steps, eps)
        results[eps] = (avg_reward, pct_optimal)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Best possible reward: E[max q*(a)] across tasks.
    # The theoretical value is ~1.539. With finite samples, we get slightly different results depending on the PRNG and
    # sample count. On the graph we get 1.55 which is the correct best possible reward for that specific set of 2000 tasks.
    best_possible = np.max(testbed.q_star[:n_runs], axis=1).mean()

    # Top plot: Average reward
    for eps, label, color in zip(epsilons, labels, colors):
        ax1.plot(results[eps][0], label=label, color=color, linewidth=0.8)
    ax1.axhline(y=best_possible, color='gray', linestyle='--', linewidth=1,
                label=f'Best possible ({best_possible:.2f})')

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average\nreward', rotation=0, labelpad=40, va='center')
    ax1.set_xlim(0, n_steps)
    ax1.set_ylim(0, 1.6)
    ax1.legend(loc='lower right')

    # Bottom plot: % Optimal action
    for eps, label, color in zip(epsilons, labels, colors):
        ax2.plot(results[eps][1], label=label, color=color, linewidth=0.8)

    ax2.set_xlabel('Steps')
    ax2.set_ylabel('% Optimal\naction', rotation=0, labelpad=40, va='center')
    ax2.set_xlim(0, n_steps)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='lower right')

    fig.suptitle('Figure 2.2: Average performance of ε-greedy action-value methods\n'
                 'on the 10-armed testbed', y=1.02)

    plt.tight_layout()
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=150)
    plt.show()


if __name__ == '__main__':
    print("Setting up 10-armed testbed...")
    tb = MultiArmedTestbed(n_arms=10, n_tasks=2000, seed=42)
    print("Generating Figure 2.2 (this may take a minute)...")
    plot_figure_2_2(tb)
    print(f"Saved {OUTPUT_FILE}")
