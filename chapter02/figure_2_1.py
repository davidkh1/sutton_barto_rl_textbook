"""
Figure 2.1: An example bandit problem from the 10-armed testbed.

Shows violin plots of reward distributions for each action,
with true values q*(a) marked.

From the RL book:
"Figure 2.1: An example bandit problem from the 10-armed testbed. The true value q⇤(a) of
each of the ten actions was selected according to a normal distribution with mean zero and unit
variance, and then the actual rewards were selected according to a mean q⇤(a), unit-variance
normal distribution, as suggested by these gray distributions."
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from multi_armed_testbed import MultiArmedTestbed

OUTPUT_FILE = Path(__file__).parent / 'output' / 'figure_2_1.png'


def plot_figure_2_1(testbed: MultiArmedTestbed, task: int = 0) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    q_star = testbed.q_star[task]

    # Generate samples for violin plot
    n_samples = 1000
    rng = np.random.default_rng(123)
    samples = [rng.normal(q_star[a], 1.0, n_samples) for a in range(testbed.n_arms)]

    parts = ax.violinplot(samples, positions=range(1, testbed.n_arms + 1),
                          showmeans=False, showmedians=False, showextrema=False)

    for pc in parts['bodies']:  # type: ignore[union-attr]
        pc.set_facecolor('gray')
        pc.set_alpha(0.7)

    # Mark true values q*(a)
    for a in range(testbed.n_arms):
        ax.plot([a + 0.7, a + 1.3], [q_star[a], q_star[a]], 'k-', linewidth=1)
        ax.text(a + 1.35, q_star[a], f'$q_*$({a+1})', fontsize=8, va='center')

    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Action')
    ax.set_ylabel('Reward\ndistribution', rotation=0, labelpad=40, va='center')
    ax.set_xticks(range(1, testbed.n_arms + 1))
    ax.set_xlim(0.5, testbed.n_arms + 1.5)
    ax.set_ylim(-4, 4)
    ax.set_title('Figure 2.1: An example bandit problem from the 10-armed testbed')

    plt.tight_layout()
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=150)
    plt.show()


if __name__ == '__main__':
    tb = MultiArmedTestbed(n_arms=10, n_tasks=2000, seed=42)
    plot_figure_2_1(tb, task=0)
    print(f"Saved {OUTPUT_FILE}")
