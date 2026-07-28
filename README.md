# Learning Quant

A collection of small, self-contained quantitative finance projects built in Python
while studying probability, statistics, and portfolio theory. Each folder is
independent — see inline docstrings/comments for details.

## Projects

### 📈 Equity Backtester (`/equity-backtester`)

Backtests moving-average crossover strategies (20/50) against passive buy-and-hold
returns on S&P 500 equities (2020–2025, daily data).

- Incorporates transaction costs and turnover
- Computes annualized Sharpe ratio and maximum drawdown
- Compares active strategy performance vs. passive benchmark

**Files:** `backtest.py`, `config.py`, `functions.py`  
**Tools:** pandas, NumPy, matplotlib

### 📊 Mean-Variance Portfolio Optimization (`/mvo`)

Implements Markowitz mean-variance optimization as a quadratic program to construct
efficient frontier portfolios.

- Formulates and solves the QP using SciPy
- Computes covariance matrices and optimal weights across risk levels
- Benchmarks efficient frontier portfolios against a 1/N (equal-weight) baseline

**Files:** `mvo.py`  
**Data:** `data/Mean-Variance Optimization.xlsx`  
**Tools:** NumPy, SciPy, pandas

### 🎲 GBM Simulation (`/gbm`)

Simple Geometric Brownian Motion path simulator — foundational stochastic process
work underpinning the other projects.

**Files:** `gbm.py`

### 🪙 Coinflip (`/coinflip`)

Small probability simulation exploring convergence and law of large numbers concepts.

**Files:** `coinflip.py`

## Shared Utilities

- `/utils` — helper functions shared across projects
- `/tests` — unit tests

## Setup

\`\`\`bash
git clone https://github.com/davidmandres/learning-quant.git
cd learning-quant
pip install -r requirements.txt
\`\`\`

Run individual scripts directly, e.g.:

\`\`\`bash
python equity-backtester/backtest.py
python -m mvo.mvo
\`\`\`

## About

Built as part of ongoing self-study in quantitative finance and probability.
See also my more extensive project:
[Quantitative Risk Engine](https://github.com/davidmandres/Risk-Engine) — GBM, GARCH, and
VaR/CVaR backtesting.
