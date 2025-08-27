import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from matplotlib.colors import to_rgba

from config import periods_per_year, R_f, tickers

def get_risk(returns: pd.Series, cum_returns: pd.Series):
    annualized_vol = returns.std() * np.sqrt(periods_per_year)
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_dd = drawdown.min()
    excess_returns = returns - R_f / periods_per_year
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year)
    hit_ratio = (returns > 0).mean()

    return {
        "annualized_vol": annualized_vol,
        "rolling_max": rolling_max,
        "drawdown": drawdown,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "hit_ratio": hit_ratio
    }

def make_plot(colors, data1: pd.Series, data2: Optional[pd.Series], comparison = True):
    for ticker in tickers:
        label = ticker if not comparison else f"{ticker} Buy & Hold"
        plt.plot(data1.index, data1[ticker], label=label, color=colors[ticker], linestyle="--")

        if comparison:
            # assert data2 != None, "data2 has a value if comparison is true" 
            plt.plot(data2.index, data2[ticker], label=f"{ticker} Strategy", color=to_rgba(colors[ticker], alpha=0.5), linestyle="-")

def add_values_to_bar_plot(bars, data):
    for bar, ticker in zip(bars, data.index):
      yval = bar.get_height()
      plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02 if yval > 0 else yval - 0.3, f"{data[ticker]:.2f}", 
              ha='center', va='bottom', fontsize=8)
