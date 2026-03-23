# Exercise 2.6: Mysterious Spikes

## Problem

> *The results shown in Figure 2.3 should be quite reliable because they are averages
> over 2000 individual, randomly chosen 10-armed bandit tasks. Why, then, are there
> oscillations and spikes in the early part of the curve for the optimistic method?
> In other words, what might make this method perform particularly better or worse,
> on average, on particular early steps?*

## Zoomed-in view of the spikes

![Mysterious spikes in the optimistic method](output/figure_2_3_spikes.png)

The first spike at step 10 is dramatic (~43%). Later peaks at steps ~22 and ~36 are
much smaller because runs desynchronize — different runs explore arms in different
orders, so the spikes blur when averaged over 2000 runs.

## Solution

The optimistic greedy method (Q1=5, epsilon=0) always picks the action with the highest
Q estimate. Since all Q values start at 5 (far above the true values ~N(0,1)), the agent
is systematically "disappointed" by every action it tries.

### Why spikes occur at specific steps

With 10 arms and greedy selection:

- **Steps 0-9**: The agent tries each arm roughly once. After trying arm `a`, its Q drops
  (e.g., from 5 to ~4.5), making untried arms (still at 5) look better. So the agent
  cycles through all 10 arms.

- **Step 10**: All arms have been tried once. Now the agent picks the arm with the
  highest Q — which is the arm that gave the best reward on its single trial. This is
  more likely to be the truly best arm, causing a **spike upward** in % optimal action.

- **Steps 11-19**: The agent re-explores. The arm it just picked at step 10 gets its Q
  pulled down further (still disappointing relative to Q~4.5). Other arms that haven't
  been tried twice still have higher Q values, so the agent cycles through them again.

- **Step ~20**: Another spike — all arms have been tried twice, and again the agent picks
  the best-looking one.

### Why later peaks are at ~22 and ~36, not 20 and 30

The first round is perfectly synchronized: every run starts with all Q=5, so all 2000 runs
try each arm exactly once in steps 0–9, then make a greedy choice at step 10.

But after step 10, runs diverge. At step 10 the agent picks arm X (highest Q). At step 11,
arm X's Q drops further and the agent switches to the next-highest. But **which arm** that
is differs across runs — each has different q* values and different reward noise. So:

- Some runs finish their second round at step 19
- Others at step 21, 22, or 23
- When averaged over 2000 runs, the peak is smeared and shifted to ~22 instead of a
  clean 20

The drift gets worse each round: by round 3, runs are even more desynchronized, pushing
the peak to ~36 instead of 30. The shift is always **later** than the ideal (10, 20, 30)
because some runs revisit the same arm twice before moving on — when reward noise makes
one arm's Q drop below another already-visited arm, effectively delaying the completion
of a round.

This is also why the later peaks are much smaller: the first spike reaches ~43% because
all runs spike together at exactly step 10, while later peaks are spread over several
steps and largely cancel out when averaged.

### The key insight

The spikes happen at multiples of ~10 steps because that's how many arms there are.
After every "round" of exploration (trying each arm once more), the agent briefly has
updated information about all arms and makes a well-informed greedy choice — producing
a momentary spike in performance. Between rounds, it's exploring arms with stale
high Q values, performing worse.

This oscillation is a signature of optimistic initialization: it creates systematic,
periodic exploration rather than the random exploration of epsilon-greedy.
