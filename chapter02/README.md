# Chapter 2

Reproductions and solutions of examples and problems from Chapter 2 of
[Reinforcement Learning: An Introduction, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html).

## The 10-Armed Testbed

- Core code: [multi_armed_testbed.py](multi_armed_testbed.py)
- Figure 2.1: [code](figure_2_1.py) and [graph](results/figure_2_1.png)
- Figure 2.2: [code](figure_2_2.py) and [graph](results/figure_2_2.png)
- Figure 2.3: Optimistic initial values — [code](figure_2_3.py) and [graph](results/figure_2_3.png)
- Best possible reward per step is ~1.54: [Monte Carlo estimate](best_reward_per_step.py) and
  [analytical derivation](best_reward_per_step_derivation.ipynb)

## Exercises

- Exercise 2.4: General weighting for non-constant step sizes — [derivation (LaTeX)](exercise_2_4.tex) and [PDF](results/exercise_2_4.pdf)
- Exercise 2.5: Nonstationary bandit problem — [code](exercise_2_5.py) and [graph](results/exercise_2_5.png)
- Exercise 2.6: [Mysterious Spikes](exercise_2_6.md) — why oscillations appear in Figure 2.3
- Exercise 2.7: Unbiased constant-step-size trick — [derivation (LaTeX)](exercise_2_7.tex) and [PDF](results/exercise_2_7.pdf)

## Interactive Simulation

- [**Interactive Bandit Testbed**](https://davidkh1.github.io/deeprl_educational/sutton_barto_rl_textbook/chapter02/interactive_bandit_testbed/) 
— browser-based simulation comparing various methods ([source](interactive_bandit_testbed/index.html)). 
For information about supported mouse operations, see the help at the bottom of the demo page.