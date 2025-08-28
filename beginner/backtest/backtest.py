import matplotlib as mpl
import yfinance as yf
import matplotlib.pyplot as plt

import sys

from config import tickers, short, long, cost_per_trade, fig_size
from functions import get_risk, make_plot, make_bar_plot 

# Step 0: Get price data
data = yf.download(tickers=tickers, start="2020-08-24", end="2025-08-24")

assert data is not None, "Failed to retrieve data"
prices = data["Close"]

# Strategy
rolling_avg_short = prices.rolling(window=short).mean()
rolling_avg_long = prices.rolling(window=long).mean()
signal = (rolling_avg_short > rolling_avg_long).astype(int)

# benchmark b, strategy s
position = signal.shift(1) # The shift avoids lookahead bias (you can’t trade today based on information you only know at the close today). By shifting, your signal today only affects your position tomorrow.
b_returns = prices.pct_change()
s_returns = b_returns * position
turnover = position.diff().abs()
net_s_returns = s_returns - cost_per_trade * turnover

# equity curve & benchmark, starting capital V_0 = 1, hence + 1
b_cum_returns = (1 + b_returns).cumprod()
s_cum_returns = (1 + net_s_returns).cumprod()

b_risk = get_risk(b_returns, b_cum_returns)
s_risk = get_risk(net_s_returns, s_cum_returns)

# sys.exit("Exit before plotting")

plt.style.use("dark_background")
mpl.rcParams['font.family'] = 'serif'

cmap = plt.get_cmap("tab10", len(tickers))
colors = {ticker: cmap(i) for i, ticker in enumerate(tickers)}

plt.figure(figsize=fig_size)
make_plot(colors, data1=prices, data2=None, comparison=False)
plt.title("Stock Prices")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)

plt.figure(figsize=fig_size)
make_plot(colors, b_cum_returns, s_cum_returns)
plt.title("Cumulative Returns")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(True)

plt.figure(figsize=fig_size)
make_plot(colors, b_risk["drawdown"], s_risk["drawdown"])
plt.title("Drawdown of tickers")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.legend()
plt.grid(True)

plt.figure(figsize=fig_size)
make_bar_plot(s_risk["sharpe"], colors)
plt.title(f"{tickers} Sharpe ratios, strategy (2020–2025)")
plt.xticks([])
plt.ylabel("Sharpe ratio")
plt.legend()
plt.grid(True)

plt.figure(figsize=fig_size)
make_bar_plot(b_risk["sharpe"], colors)
plt.title(f"{tickers} Sharpe ratios, benchmark (2020–2025)")
plt.xticks([])
plt.ylabel("Sharpe ratio")
plt.legend()
plt.grid(True)

plt.show()
