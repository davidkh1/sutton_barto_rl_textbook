# Exercise 2.10: Associative Search (Contextual Bandits)

## Problem

> *Suppose you face a 2-armed bandit task whose true action values change randomly
> from time step to time step. Specifically, suppose that, for any time step, the true
> values of actions 1 and 2 are respectively 0.1 and 0.2 with probability 0.5 (case A),
> and 0.9 and 0.8 with probability 0.5 (case B). If you are not able to tell which case
> you face at any step, what is the best expected reward you can achieve and how should
> you behave to achieve it? Now suppose that on each step you are told whether you are
> facing case A or case B (although you still don't know the true action values). This
> is an associative search task. What is the best expected reward you can achieve in
> this task, and how should you behave to achieve it?*

## Solution

### Part 1: No context (nonassociative)

Without knowing which case you face, the expected value of each action is:

- E[q(action 1)] = 0.5 × 0.1 + 0.5 × 0.9 = **0.50**
- E[q(action 2)] = 0.5 × 0.2 + 0.5 × 0.8 = **0.50**

Both actions have the same expected value! Any policy (always pick 1, always pick 2,
alternate, randomize) yields the same expected reward:

**Best expected reward = 0.50**

**Can't we infer the case from past rewards?** One might think: "If I observe a high
reward (~0.8–0.9), I'm probably in case B, so I should prefer action 1 next time."
But this doesn't help — the case is drawn **independently** each step (i.i.d.). A high
reward on step t tells you nothing about which case you'll face on step t+1. Since
you must choose your action *before* observing the current step's reward, there is no
way to exploit within-step information either.

### Part 2: With context (associative search)

Now you are told whether it's case A or case B **before** choosing. You can learn a
separate policy for each case:

- **Case A** (q = [0.1, 0.2]): pick action 2 → reward = **0.2**
- **Case B** (q = [0.9, 0.8]): pick action 1 → reward = **0.9**

Each case occurs with probability 0.5, so:

**Best expected reward = 0.5 × 0.2 + 0.5 × 0.9 = 0.55**

### How to behave

You don't know the true values initially — you only know which case you're in.
Use any standard bandit method (ε-greedy, UCB, etc.) but maintain **separate
Q estimates for each case**. Over time, you'll learn:

- In case A: Q(action 2) > Q(action 1)
- In case B: Q(action 1) > Q(action 2)

### The key insight

Context improves the best possible reward from 0.50 to 0.55 — a 10% gain. Without
context, you can't distinguish the cases and both actions look equally good.
With context, you can exploit the structure: always pick the better action for
each case.

This illustrates why associative search (contextual bandits) is more powerful than
the nonassociative setting. It's a stepping stone toward full reinforcement learning,
where the agent's actions also affect the next state.
