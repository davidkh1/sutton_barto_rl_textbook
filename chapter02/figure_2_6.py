"""
Figure 2.6: A parameter study of the various bandit algorithms.

Each point is the average reward obtained over first 1000 steps with a particular
algorithm at a particular setting of its parameter. All algorithms use sample
averages except optimistic greedy which uses constant α=0.1.

Algorithms and their varied parameters:
  - ε-greedy: ε ∈ {1/128, 1/64, ..., 1/4}
  - Gradient bandit: α ∈ {1/32, 1/16, ..., 2}
  - UCB: c ∈ {1/16, 1/8, ..., 4}
  - Greedy with optimistic initialization: Q₀ ∈ {1/4, 1/2, ..., 4}, α=0.1

From the RL book:
"Figure 2.6: A parameter study of the various bandit algorithms presented in this
chapter. Each point is the average reward obtained over 1000 steps with a particular
algorithm at a particular setting of its parameter."
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from multi_armed_testbed import MultiArmedTestbed

OUTPUT_FILE = Path(__file__).parent / 'output' / 'figure_2_6.png'

N_RUNS = 2000
N_STEPS = 1000
N_ARMS = 10
SEED = 42


def run_eps_greedy(q_star, n_steps, epsilon, seed):
    """ε-greedy with sample averages."""
    n_runs, n_arms = q_star.shape
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_runs, n_arms))
    N = np.zeros((n_runs, n_arms), dtype=int)
    total_rewards = np.zeros(n_runs)
    run_idx = np.arange(n_runs)

    for t in range(n_steps):
        explore = rng.random(n_runs) < epsilon
        random_actions = rng.integers(n_arms, size=n_runs)
        noisy_q = Q + rng.random((n_runs, n_arms)) * 1e-10
        greedy_actions = np.argmax(noisy_q, axis=1)
        actions = np.where(explore, random_actions, greedy_actions)

        rewards = rng.normal(q_star[run_idx, actions], 1.0)
        N[run_idx, actions] += 1
        Q[run_idx, actions] += (rewards - Q[run_idx, actions]) / N[run_idx, actions]
        total_rewards += rewards

    return total_rewards.mean() / n_steps


def run_ucb(q_star, n_steps, c, seed):
    """UCB with sample averages."""
    n_runs, n_arms = q_star.shape
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_runs, n_arms))
    N = np.zeros((n_runs, n_arms), dtype=int)
    total_rewards = np.zeros(n_runs)
    run_idx = np.arange(n_runs)

    for t in range(n_steps):
        safe_N = np.maximum(N, 1)
        ucb_bonus = c * np.sqrt(np.log(t + 1) / safe_N)
        ucb_bonus[N == 0] = np.inf
        ucb_values = Q + ucb_bonus + rng.random((n_runs, n_arms)) * 1e-10
        actions = np.argmax(ucb_values, axis=1)

        rewards = rng.normal(q_star[run_idx, actions], 1.0)
        N[run_idx, actions] += 1
        Q[run_idx, actions] += (rewards - Q[run_idx, actions]) / N[run_idx, actions]
        total_rewards += rewards

    return total_rewards.mean() / n_steps


def run_gradient(q_star, n_steps, alpha, seed):
    """Gradient bandit with baseline."""
    n_runs, n_arms = q_star.shape
    rng = np.random.default_rng(seed)
    H = np.zeros((n_runs, n_arms))
    avg_reward = np.zeros(n_runs)
    total_rewards = np.zeros(n_runs)
    run_idx = np.arange(n_runs)

    for t in range(n_steps):
        H_stable = H - H.max(axis=1, keepdims=True)
        exp_H = np.exp(H_stable)
        pi = exp_H / exp_H.sum(axis=1, keepdims=True)

        cumulative = pi.cumsum(axis=1)
        random_vals = rng.random((n_runs, 1))
        actions = (random_vals >= cumulative).sum(axis=1)
        actions = np.clip(actions, 0, n_arms - 1)

        rewards = rng.normal(q_star[run_idx, actions], 1.0)

        one_hot = np.zeros((n_runs, n_arms))
        one_hot[run_idx, actions] = 1.0
        H += alpha * (rewards - avg_reward)[:, np.newaxis] * (one_hot - pi)

        avg_reward += (rewards - avg_reward) / (t + 1)
        total_rewards += rewards

    return total_rewards.mean() / n_steps


def run_optimistic_greedy(q_star, n_steps, q_init, alpha, seed):
    """Greedy with optimistic initialization and constant step-size α."""
    n_runs, n_arms = q_star.shape
    rng = np.random.default_rng(seed)
    Q = np.full((n_runs, n_arms), q_init)
    total_rewards = np.zeros(n_runs)
    run_idx = np.arange(n_runs)

    for t in range(n_steps):
        noisy_q = Q + rng.random((n_runs, n_arms)) * 1e-10
        actions = np.argmax(noisy_q, axis=1)

        rewards = rng.normal(q_star[run_idx, actions], 1.0)
        Q[run_idx, actions] += alpha * (rewards - Q[run_idx, actions])
        total_rewards += rewards

    return total_rewards.mean() / n_steps


def plot_figure_2_6(all_results: dict[str, list[tuple[float, float]]]) -> None:
    """Parameter study plot with log-scale x-axis."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {
        'ε-greedy': 'red',
        'gradient bandit': 'green',
        'UCB': 'blue',
        'greedy with optimistic\ninitialization α=0.1': 'black',
    }

    for name, points in all_results.items():
        params, rewards = zip(*points)
        ax.plot(params, rewards, marker='o', markersize=4, label=name,
                color=colors[name], linewidth=1.5)

    ax.set_xscale('log', base=2)
    ax.set_ylabel('Average\nreward\nover first\n1000 steps',
                  rotation=0, labelpad=50, va='center')
    ax.set_xlim(1/256, 8)
    ax.set_ylim(1.0, 1.55)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Figure 2.6: Parameter study of bandit algorithms on the 10-armed testbed')

    # Custom x-tick labels
    ticks = [2**i for i in range(-7, 3)]
    tick_labels = ['1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4']
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)

    # Color-coded parameter labels on x-axis
    ax.set_xlabel('')  # clear default
    label_data = [
        ('ε', 'red'),
        ('α', 'green'),
        ('c', 'blue'),
        ('$Q_0$', 'black'),
    ]
    x_pos = 0.35
    for label_text, color in label_data:
        ax.annotate(label_text, xy=(x_pos, -0.13), xycoords='axes fraction',
                    fontsize=13, fontweight='bold', color=color,
                    ha='center', va='top')
        x_pos += 0.1

    plt.tight_layout()
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=150)
    plt.show()


if __name__ == '__main__':
    testbed = MultiArmedTestbed(n_arms=N_ARMS, n_tasks=N_RUNS, seed=SEED)
    q_star = testbed.q_star
    sim_seed = SEED + 1

    # ε-greedy: ε from 1/128 to 1/4
    print("Running ε-greedy...")
    eps_results = []
    for exp in range(-7, -1):
        eps = 2.0 ** exp
        reward = run_eps_greedy(q_star, N_STEPS, eps, sim_seed)
        eps_results.append((eps, reward))
        print(f"  ε={eps:.4f}: {reward:.3f}")

    # Gradient bandit: α from 1/32 to 2
    print("Running gradient bandit...")
    grad_results = []
    for exp in range(-5, 2):
        alpha = 2.0 ** exp
        reward = run_gradient(q_star, N_STEPS, alpha, sim_seed)
        grad_results.append((alpha, reward))
        print(f"  α={alpha:.4f}: {reward:.3f}")

    # UCB: c from 1/16 to 4
    print("Running UCB...")
    ucb_results = []
    for exp in range(-4, 3):
        c = 2.0 ** exp
        reward = run_ucb(q_star, N_STEPS, c, sim_seed)
        ucb_results.append((c, reward))
        print(f"  c={c:.4f}: {reward:.3f}")

    # Optimistic greedy: Q_0 from 1/4 to 4, α=0.1
    print("Running optimistic greedy...")
    opt_results = []
    for exp in range(-2, 3):
        q_init = 2.0 ** exp
        reward = run_optimistic_greedy(q_star, N_STEPS, q_init, 0.1, sim_seed)
        opt_results.append((q_init, reward))
        print(f"  Q₀={q_init:.4f}: {reward:.3f}")

    experiment_results = {
        'ε-greedy': eps_results,
        'gradient bandit': grad_results,
        'UCB': ucb_results,
        'greedy with optimistic\ninitialization α=0.1': opt_results,
    }

    plot_figure_2_6(experiment_results)
    print(f"Saved {OUTPUT_FILE}")
