# Exercise 2.8: UCB Spikes

## Problem

> *In Figure 2.4 the UCB algorithm shows a distinct spike in performance on the 11th
> step. Why is this? Note that for your answer to be fully satisfactory it must explain
> both why the reward increases on the 11th step and why it decreases on the subsequent
> steps. Hint: If c = 1, then the spike is less prominent.*

## Solution

### Why the spike occurs on step 11

With 10 arms and UCB action selection, untried actions have infinite bonus
(N_t(a) = 0 makes the UCB term infinite). So:

- **Steps 1-10**: UCB selects each untried arm exactly once, in a random order
  (ties among infinite bonuses are broken randomly). These are essentially random
  selections — the average reward is E[q*(a)] = 0 across all arms.

- **Step 11**: All arms have been tried once. Now UCB makes its first *informed* decision
  based on actual Q estimates and uncertainty bonuses. The action selected is:

  A_11 = argmax_a [Q(a) + c · sqrt(ln(11) / 1)]

  Since every arm has been tried once (N(a) = 1 for all a), the bonus term
  c · sqrt(ln(11)) is **identical** for all arms. It cancels out! The selection
  reduces to simply picking the arm with the highest Q estimate — which is the arm
  that gave the highest single reward. This arm is more likely to be the truly best
  arm, causing the spike in average reward.

This is the same mechanism as the step-10 spike in Figure 2.3 (Exercise 2.6): after
one complete round of exploration, the agent makes a well-informed greedy choice.

### Why the reward *decreases* after step 11

On step 12, the arm chosen at step 11 now has N(a) = 2 while all others still have
N(a) = 1. The UCB values become:

- For the step-11 arm: Q(a) + c · sqrt(ln(12) / 2)
- For all other arms: Q(a) + c · sqrt(ln(12) / 1)

The other arms get a **larger uncertainty bonus** (sqrt(ln(12)) vs sqrt(ln(12)/2)).
With c = 2, this bonus is substantial: 2 · sqrt(ln(12)) ≈ 3.15 vs 2 · sqrt(ln(12)/2) ≈ 2.23.
This ~0.9 difference is large enough to pull UCB toward a less-tried (but possibly worse) arm.

UCB is designed to do this — it systematically re-explores arms that haven't been tried
recently. But in the short term, this means stepping away from the best-looking arm to
gather more information, temporarily lowering the average reward.

### Why the spike is less prominent when c = 1

The hint says c = 1 makes the spike less prominent. With a smaller c:

- The uncertainty bonus is smaller, so the "pull" toward untried arms is weaker
- On step 11, even though all bonuses are equal, the preceding steps 1-10 still
  select untried arms (infinite bonus), so the behavior is the same
- But the *contrast* is smaller: with c = 1 the post-spike exploration is less
  aggressive, so the reward doesn't drop as sharply after step 11
- The spike height (step 11 reward minus surrounding steps) is smaller because
  the exploration penalty on steps 12+ is milder

In the limit c → 0, UCB becomes greedy and there would be no spike at all (but also
no systematic exploration).

### The key insight

The UCB spike is analogous to the optimistic initialization spike (Exercise 2.6):
both arise from a forced exploration phase (first 10 steps) followed by a single
well-informed choice. The difference is that UCB's exploration is driven by the
uncertainty bonus rather than by disappointment from inflated Q values. After step 11,
UCB continues to explore systematically (favoring less-tried arms), which temporarily
reduces performance but leads to better long-term behavior than ε-greedy.
