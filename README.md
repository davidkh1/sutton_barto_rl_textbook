# "Reinforcement Learning: An Introduction" book

I created this file while studying
[RL: An Introduction, 2e](http://incompleteideas.net/book/the-book-2nd.html) by Sutton & Barto.

I believe that the Deep Reinforcement Learning is the foundation of the forthcoming
intelligent robotics revolution. I decided to fill in the gaps in my knowledge of the theory.
This directory is the result of my serious efforts to study the RL book, by developing
all the math, reproducing the graphs, and experimenting with the code.

## Setup
The repository contains code, notes, and interactive demos.
To run the code, you can use the environment that I created.
There is also a script called `split_pdf_chapters.py` that extracts the desired
chapters from the RL book and places them in the `docs` directory.

### 1. Conda setup

```bash
conda env create -f environment.yml
conda activate rl_textbook
```

### 2. Download and Split Textbook PDF

To download the textbook PDF and split it into per-chapter files, run:

```bash
python scripts/split_rlbook_to_chapters.py
```

This creates `chapterXX/docs/chXX_<title>.pdf` for each of the 17 chapters.

Options:
- `--dry-run` - preview without writing files
- `--chapters 2,3,4` - extract specific chapters only
- `--force-download` - re-download PDF even if cached

## Chapters

- [Chapter 1 — Introduction](chapter01/README.md)
- [Chapter 2 — Multi-armed Bandits](chapter02/README.md)
- [Chapter 3 — Finite Markov Decision Processes](chapter03/README.md)
- [Chapter 4 — Dynamic Programming](chapter04/README.md)
- [Chapter 5 — Monte Carlo Methods](chapter05/README.md)
- [Chapter 6 — Temporal-Difference Learning](chapter06/README.md)
- [Chapter 7 — n-step Bootstrapping](chapter07/README.md)
- [Chapter 8 — Planning and Learning with Tabular Methods](chapter08/README.md)
- [Chapter 9 — On-policy Prediction with Approximation](chapter09/README.md)
- [Chapter 10 — On-policy Control with Approximation](chapter10/README.md)
- [Chapter 11 — *Off-policy Methods with Approximation](chapter11/README.md)
- [Chapter 12 — Eligibility Traces](chapter12/README.md)
- [Chapter 13 — Policy Gradient Methods](chapter13/README.md)
- [Chapter 14 — Psychology](chapter14/README.md)
- [Chapter 15 — Neuroscience](chapter15/README.md)
- [Chapter 16 — Applications and Case Studies](chapter16/README.md)
- [Chapter 17 — Frontiers](chapter17/README.md)

## Interactive Demos

- [**Multi-Armed Bandit Testbed**](https://davidkh1.github.io/sutton_barto_rl_textbook/chapter02/interactive_bandit_testbed/) (Chapter 2) — compare Greedy, ε-Greedy, Optimistic, UCB, and Gradient Bandit methods
