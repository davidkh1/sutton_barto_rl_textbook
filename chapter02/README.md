# Chapter 2 — Multi-armed Bandits

Reproductions and solutions of examples and problems from Chapter 2 of
[RL: An Introduction, 2e](http://incompleteideas.net/book/the-book-2nd.html).

## The 10-Armed Testbed

- Core code: [multi_armed_testbed.py](multi_armed_testbed.py)
- Figure 2.1: [code](figure_2_1.py) and [graph](results/figure_2_1.png)
- Figure 2.2: [code](figure_2_2.py) and [graph](results/figure_2_2.png)
- Figure 2.3: Optimistic initial values — [code](figure_2_3.py) and [graph](results/figure_2_3.png)
- Figure 2.4: UCB action selection — [code](figure_2_4.py) and [graph](results/figure_2_4.png)
- Figure 2.5: Gradient bandit with/without baseline — [code](figure_2_5.py) and [graph](results/figure_2_5.png)
- Figure 2.6: Parameter study of all algorithms — [code](figure_2_6.py) and [graph](results/figure_2_6.png)
- Best possible reward per step is ~1.54: [Monte Carlo estimate](best_reward_per_step.py) and
  [analytical derivation](best_reward_per_step_derivation.ipynb)

## Exercises

- Exercise 2.4: General weighting for non-constant step sizes — [derivation (LaTeX)](latex/exercise_2_4.tex) and [PDF](results/exercise_2_4.pdf)
- Exercise 2.5: Nonstationary bandit problem — [code](exercise_2_5.py) and [graph](results/exercise_2_5.png)
- Exercise 2.6: [Mysterious Spikes](exercise_2_6.md) — why oscillations appear in Figure 2.3
- Exercise 2.7: Unbiased constant-step-size trick — [derivation (LaTeX)](latex/exercise_2_7.tex) and [PDF](results/exercise_2_7.pdf)
- Exercise 2.8: [UCB Spikes](exercise_2_8.md) — why the spike appears on step 11 in Figure 2.4
- Exercise 2.9: Soft-max is the sigmoid for two actions — [derivation (LaTeX)](latex/exercise_2_9.tex) and [PDF](results/exercise_2_9.pdf)
- Exercise 2.10: [Associative Search](exercise_2_10.md) — contextual bandit with two cases
- Exercise 2.11: Nonstationary parameter study — [code](exercise_2_11.py) and [graph](results/exercise_2_11.png)

## Interactive Simulation

- [**Interactive Bandit Testbed**](https://davidkh1.github.io/sutton_barto_rl_textbook/chapter02/interactive_bandit_testbed/) 
— browser-based simulation comparing various methods ([source](interactive_bandit_testbed/index.html)). 
For information about supported mouse operations, see the help at the bottom of the demo page.