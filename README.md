# "Reinforcement Learning: An Introduction" book

I created this file while studying
[Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) by Sutton & Barto (2nd edition).

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
python scripts/split_pdf_chapters.py
```

This creates `chapterXX/docs/chXX_<title>.pdf` for each of the 17 chapters.

Options:
- `--dry-run` - preview without writing files
- `--chapters 2,3,4` - extract specific chapters only
- `--force-download` - re-download PDF even if cached

## Interactive Demos

- [**Multi-Armed Bandit Testbed**](https://davidkh1.github.io/deeprl_educational/sutton_barto_rl_textbook/chapter02/interactive_bandit_testbed/) (Chapter 2) — compare Greedy, ε-Greedy, Optimistic, UCB, and Gradient Bandit methods
