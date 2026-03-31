"""
Exercise 2.11: Parameter study for the nonstationary case.

Makes a figure analogous to Figure 2.6 for the nonstationary case outlined in
Exercise 2.5. Includes the constant-step-size ε-greedy algorithm with α=0.1.
Uses runs of 200,000 steps, with performance measured as the average reward
over the last 100,000 steps.

The nonstationary testbed: all q*(a) start equal (at 0), then take independent
random walks (std=0.01) each step.

Algorithms compared:
  - ε-greedy with sample averages (1/n step-size)
  - ε-greedy with constant α=0.1
  - UCB with sample averages
  - Gradient bandit with baseline
  - Greedy with optimistic initialization, α=0.1

Key findings (compare with Figure 2.6 for the stationary case):
  - ε-greedy with constant α=0.1 dominates all other methods. The constant step-size
    forgets old data exponentially, allowing Q estimates to track the drifting q* values.
  - ε-greedy with sample averages (1/n) is much worse — the step-size shrinks to zero,
    so after many steps the estimates barely update and go stale.
  - UCB performs poorly despite being the best in the stationary case. Its sample-average
    estimates go stale, and the ln(t)/N(a) bonus shrinks over time, reducing exploration
    exactly when the nonstationary environment demands more of it.
  - Gradient bandit also struggles — its baseline (average of all past rewards) is slow
    to adapt, and the preference updates accumulate stale information.
  - Optimistic greedy is mediocre — the constant α=0.1 helps it track, but pure greedy
    (ε=0) means it rarely re-explores arms whose q* may have improved.
  - The ranking is completely flipped vs the stationary case: methods that forget old
    data (constant α) beat methods that remember everything (sample averages).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_FILE = Path(__file__).parent / 'output' / 'exercise_2_11.png'

N_RUNS = 200
N_STEPS = 200_000
LAST_STEPS = 100_000
N_ARMS = 10
WALK_STD = 0.01


def run_eps_greedy_sa(n_runs, n_arms, n_steps, epsilon, walk_std, seed):
    """ε-greedy with sample averages (1/n) on nonstationary testbed."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_runs, n_arms))
    N = np.zeros((n_runs, n_arms), dtype=int)
    q_star = np.zeros((n_runs, n_arms))
    total_reward = 0.0
    run_idx = np.arange(n_runs)
    measure_from = n_steps - LAST_STEPS

    for t in range(n_steps):
        # Action selection
        explore = rng.random(n_runs) < epsilon
        random_actions = rng.integers(n_arms, size=n_runs)
        noisy_q = Q + rng.random((n_runs, n_arms)) * 1e-10
        actions = np.where(explore, random_actions, np.argmax(noisy_q, axis=1))

        # Rewards and update
        rewards = rng.normal(q_star[run_idx, actions], 1.0)
        N[run_idx, actions] += 1
        Q[run_idx, actions] += (rewards - Q[run_idx, actions]) / N[run_idx, actions]

        if t >= measure_from:
            total_reward += rewards.sum()

        # Random walk
        q_star += rng.normal(0, walk_std, (n_runs, n_arms))

    return total_reward / (LAST_STEPS * n_runs)


def run_eps_greedy_const(n_runs, n_arms, n_steps, epsilon, alpha, walk_std, seed):
    """ε-greedy with constant step-size α on nonstationary testbed."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_runs, n_arms))
    q_star = np.zeros((n_runs, n_arms))
    total_reward = 0.0
    run_idx = np.arange(n_runs)
    measure_from = n_steps - LAST_STEPS

    for t in range(n_steps):
        explore = rng.random(n_runs) < epsilon
        random_actions = rng.integers(n_arms, size=n_runs)
        noisy_q = Q + rng.random((n_runs, n_arms)) * 1e-10
        actions = np.where(explore, random_actions, np.argmax(noisy_q, axis=1))

        rewards = rng.normal(q_star[run_idx, actions], 1.0)
        Q[run_idx, actions] += alpha * (rewards - Q[run_idx, actions])

        if t >= measure_from:
            total_reward += rewards.sum()

        q_star += rng.normal(0, walk_std, (n_runs, n_arms))

    return total_reward / (LAST_STEPS * n_runs)


def run_ucb(n_runs, n_arms, n_steps, c, walk_std, seed):
    """UCB with sample averages on nonstationary testbed."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_runs, n_arms))
    N = np.zeros((n_runs, n_arms), dtype=int)
    q_star = np.zeros((n_runs, n_arms))
    total_reward = 0.0
    run_idx = np.arange(n_runs)
    measure_from = n_steps - LAST_STEPS

    for t in range(n_steps):
        safe_N = np.maximum(N, 1)
        ucb_bonus = c * np.sqrt(np.log(t + 1) / safe_N)
        ucb_bonus[N == 0] = np.inf
        ucb_values = Q + ucb_bonus + rng.random((n_runs, n_arms)) * 1e-10
        actions = np.argmax(ucb_values, axis=1)

        rewards = rng.normal(q_star[run_idx, actions], 1.0)
        N[run_idx, actions] += 1
        Q[run_idx, actions] += (rewards - Q[run_idx, actions]) / N[run_idx, actions]

        if t >= measure_from:
            total_reward += rewards.sum()

        q_star += rng.normal(0, walk_std, (n_runs, n_arms))

    return total_reward / (LAST_STEPS * n_runs)


def run_gradient(n_runs, n_arms, n_steps, alpha, walk_std, seed):
    """Gradient bandit with baseline on nonstationary testbed."""
    rng = np.random.default_rng(seed)
    H = np.zeros((n_runs, n_arms))
    avg_reward = np.zeros(n_runs)
    q_star = np.zeros((n_runs, n_arms))
    total_reward = 0.0
    run_idx = np.arange(n_runs)
    measure_from = n_steps - LAST_STEPS

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

        if t >= measure_from:
            total_reward += rewards.sum()

        q_star += rng.normal(0, walk_std, (n_runs, n_arms))

    return total_reward / (LAST_STEPS * n_runs)


def run_optimistic_greedy(n_runs, n_arms, n_steps, q_init, alpha, walk_std, seed):
    """Greedy with optimistic initialization and constant α on nonstationary testbed."""
    rng = np.random.default_rng(seed)
    Q = np.full((n_runs, n_arms), q_init)
    q_star = np.zeros((n_runs, n_arms))
    total_reward = 0.0
    run_idx = np.arange(n_runs)
    measure_from = n_steps - LAST_STEPS

    for t in range(n_steps):
        noisy_q = Q + rng.random((n_runs, n_arms)) * 1e-10
        actions = np.argmax(noisy_q, axis=1)

        rewards = rng.normal(q_star[run_idx, actions], 1.0)
        Q[run_idx, actions] += alpha * (rewards - Q[run_idx, actions])

        if t >= measure_from:
            total_reward += rewards.sum()

        q_star += rng.normal(0, walk_std, (n_runs, n_arms))

    return total_reward / (LAST_STEPS * n_runs)


def plot_exercise_2_11(all_results: dict[str, list[tuple[float, float]]]) -> None:
    """Parameter study plot for nonstationary case."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {
        'ε-greedy (sample avg)': 'red',
        'ε-greedy (α=0.1)': 'darkred',
        'gradient bandit': 'green',
        'UCB': 'blue',
        'optimistic greedy α=0.1': 'black',
    }
    linestyles = {
        'ε-greedy (sample avg)': '--',
        'ε-greedy (α=0.1)': '-',
        'gradient bandit': '-',
        'UCB': '-',
        'optimistic greedy α=0.1': '-',
    }

    for name, points in all_results.items():
        params, rewards = zip(*points)
        ax.plot(params, rewards, marker='o', markersize=4, label=name,
                color=colors[name], linestyle=linestyles[name], linewidth=1.5)

    ax.set_xscale('log', base=2)
    ax.set_ylabel('Average reward\nover last\n100,000 steps',
                  rotation=0, labelpad=60, va='center')
    ax.set_xlim(1/256, 8)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Exercise 2.11: Parameter study — nonstationary case\n'
                 f'({N_RUNS} runs × {N_STEPS:,} steps, walk σ={WALK_STD})')

    ticks = [2**i for i in range(-7, 3)]
    tick_labels = ['1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4']
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)

    # Color-coded parameter labels
    ax.set_xlabel('')
    label_data = [
        ('ε', 'red'), ('ε', 'darkred'), ('α', 'green'), ('c', 'blue'), ('$Q_0$', 'black'),
    ]
    x_pos = 0.25
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
    import time
    sim_seed = 43

    # ε-greedy with sample averages
    print("Running ε-greedy (sample avg)...")
    eps_sa_results = []
    for exp in range(-7, -1):
        eps = 2.0 ** exp
        t0 = time.time()
        reward = run_eps_greedy_sa(N_RUNS, N_ARMS, N_STEPS, eps, WALK_STD, sim_seed)
        eps_sa_results.append((eps, reward))
        print(f"  ε={eps:.4f}: {reward:.3f}  ({time.time() - t0:.0f}s)")

    # ε-greedy with constant α=0.1
    print("Running ε-greedy (α=0.1)...")
    eps_const_results = []
    for exp in range(-7, -1):
        eps = 2.0 ** exp
        t0 = time.time()
        reward = run_eps_greedy_const(N_RUNS, N_ARMS, N_STEPS, eps, 0.1, WALK_STD, sim_seed)
        eps_const_results.append((eps, reward))
        print(f"  ε={eps:.4f}: {reward:.3f}  ({time.time() - t0:.0f}s)")

    # Gradient bandit
    print("Running gradient bandit...")
    grad_results = []
    for exp in range(-5, 2):
        alpha = 2.0 ** exp
        t0 = time.time()
        reward = run_gradient(N_RUNS, N_ARMS, N_STEPS, alpha, WALK_STD, sim_seed)
        grad_results.append((alpha, reward))
        print(f"  α={alpha:.4f}: {reward:.3f}  ({time.time() - t0:.0f}s)")

    # UCB
    print("Running UCB...")
    ucb_results = []
    for exp in range(-4, 3):
        c = 2.0 ** exp
        t0 = time.time()
        reward = run_ucb(N_RUNS, N_ARMS, N_STEPS, c, WALK_STD, sim_seed)
        ucb_results.append((c, reward))
        print(f"  c={c:.4f}: {reward:.3f}  ({time.time() - t0:.0f}s)")

    # Optimistic greedy
    print("Running optimistic greedy...")
    opt_results = []
    for exp in range(-2, 3):
        q_init = 2.0 ** exp
        t0 = time.time()
        reward = run_optimistic_greedy(N_RUNS, N_ARMS, N_STEPS, q_init, 0.1, WALK_STD, sim_seed)
        opt_results.append((q_init, reward))
        print(f"  Q₀={q_init:.4f}: {reward:.3f}  ({time.time() - t0:.0f}s)")

    experiment_results = {
        'ε-greedy (sample avg)': eps_sa_results,
        'ε-greedy (α=0.1)': eps_const_results,
        'gradient bandit': grad_results,
        'UCB': ucb_results,
        'optimistic greedy α=0.1': opt_results,
    }

    plot_exercise_2_11(experiment_results)
    print(f"Saved {OUTPUT_FILE}")
