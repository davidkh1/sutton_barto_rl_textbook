# Reinforcement Learning: An Introduction

Studying [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) by Sutton & Barto (2nd edition).

## Interactive Demos

- [**Multi-Armed Bandit Testbed**](https://davidkh1.github.io/deeprl_educational/sutton_barto_rl_textbook/chapter02/interactive_bandit_testbed/) (Chapter 2) — compare Greedy, ε-Greedy, Optimistic, UCB, and Gradient Bandit methods

## Setup

```bash
conda env create -f environment.yml
conda activate rl_textbook
```

## Download and Split Textbook PDF

Download the textbook PDF and split it into per-chapter files:

```bash
python scripts/split_pdf_chapters.py
```

This creates `chapterXX/docs/chXX_<title>.pdf` for each of the 17 chapters.

Options:
- `--dry-run` - preview without writing files
- `--chapters 2,3,4` - extract specific chapters only
- `--force-download` - re-download PDF even if cached

