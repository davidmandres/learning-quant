import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize

import matplotlib as mpl
from matplotlib import pyplot as plt

from helpers import bar_chart_nums

N = DAILY = 252
r_f = 0.01
r_prem = 0.06


EQUITY_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
# RF_TICKER = "IEF"  # 7-10 year treasury ETF as a proxy for risk-free asset
MARKET_TICKER = "SPY"

TICKERS = EQUITY_TICKERS + [MARKET_TICKER]
TICKERS_WITH_RF = EQUITY_TICKERS + ["RF"]

# mean-variance optimization (MVO) example
data = yf.download(tickers=TICKERS, period="1y", interval="1d")
assert data is not None, "No data fetched. Check ticker symbols and internet connection."
prices = data['Close']

# calculate daily returns, annualized returns and volatilities
daily_returns = prices.pct_change().dropna()
daily_std = daily_returns.std()
annualized_returns = daily_returns.mean() * DAILY
annualized_std = daily_std * np.sqrt(DAILY)

# CAPM expected returns
cov_matrix = daily_returns.cov() * DAILY
ticker_betas = cov_matrix["SPY"] / cov_matrix.loc["SPY", "SPY"]
cov_matrix = cov_matrix.drop(columns=["SPY"], index=["SPY"])  # remove market from covariance matrix
cov_matrix_rf = cov_matrix.copy()
cov_matrix_rf["RF"] = 0.0
cov_matrix_rf.loc["RF"] = 0.0
capm_returns = r_f + ticker_betas * r_prem

# simplified black-litterman expected returns, take the calculated (from the prices) expected returns and the capm returns, with equal weights
bl_expected_returns = 0.5 * (annualized_returns + capm_returns)
bl_expected_returns = bl_expected_returns[EQUITY_TICKERS]  # only equities
bl_expected_returns_rf = bl_expected_returns.copy()
bl_expected_returns_rf["RF"] = r_f  # risk-free asset expected return

def portfolio_volatility(weights, cov_matrix=cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def portfolio_return(weights, bl_expected_returns=bl_expected_returns, _cov_matrix=cov_matrix):
        return np.dot(weights, bl_expected_returns)

def sharpe_ratio(weights, bl_expected_returns=bl_expected_returns, cov_matrix=cov_matrix):
        port_return = portfolio_return(weights, bl_expected_returns)
        port_volatility = portfolio_volatility(weights, cov_matrix)
        return (port_return - r_f) / port_volatility

def portfolio_stats(weights, bl_expected_returns=bl_expected_returns, cov_matrix=cov_matrix):
        port_return = portfolio_return(weights, bl_expected_returns)
        port_volatility = portfolio_volatility(weights, cov_matrix)
        port_sharpe = sharpe_ratio(weights, bl_expected_returns, cov_matrix)
        index = ["Weights", "E(R)", "Volatility", "Sharpe Ratio"]
        return pd.Series([weights, port_return, port_volatility, port_sharpe], index=index)

def maximize_stat(stat_fun, assets = EQUITY_TICKERS, cov_matrix=cov_matrix, bl_expected_returns=bl_expected_returns, target_volatility = None):
    num_assets = len(assets)
    
    def neg(weights):
        return -stat_fun(weights, bl_expected_returns, cov_matrix) # since scipy only has minimization, we negate

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    if target_volatility is not None:
        constraints = (
            constraints,
            {'type': 'eq', 'fun': lambda x: portfolio_volatility(x, cov_matrix) - target_volatility}
        )

    bounds = tuple((-0.9, 0.9) for _ in range(num_assets)) # no more than 90% in one asset, shorting allowed

    # initial guess is equal weights
    initial_guess = np.array([1 / num_assets] * num_assets)
    
    result = minimize(neg, x0=initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    return pd.Series(result.x, index=assets)

equal_portfolio_weights = pd.Series(1 / len(EQUITY_TICKERS), index=EQUITY_TICKERS)
equal_portfolio_stats = portfolio_stats(equal_portfolio_weights)

optimized_weights = maximize_stat(portfolio_return, target_volatility=equal_portfolio_stats["Volatility"])
optimized_portfolio_stats = portfolio_stats(optimized_weights)

tangency_portfolio_weights = maximize_stat(sharpe_ratio)
tangency_portfolio_stats = portfolio_stats(tangency_portfolio_weights)

tangency_rf_portfolio_weights = maximize_stat(sharpe_ratio, assets=TICKERS_WITH_RF, cov_matrix=cov_matrix_rf, bl_expected_returns=bl_expected_returns_rf)
tangency_rf_portfolio_stats = portfolio_stats(tangency_rf_portfolio_weights, bl_expected_returns=bl_expected_returns_rf, cov_matrix=cov_matrix_rf)

# Plotting the efficient frontier
target_volatilities = np.linspace(0.1 * equal_portfolio_stats["Volatility"], 1.5 * equal_portfolio_stats["Volatility"], 50)
target_returns = []

for vol in target_volatilities:
    weights = maximize_stat(portfolio_return, target_volatility=vol)
    ret = portfolio_return(weights)
    target_returns.append(ret)

plt.style.use("dark_background")
mpl.rcParams['font.family'] = 'serif'

_, axes = plt.subplots(2, 2, figsize=(18, 9))

# 1. Efficient frontier plot
axes[0, 0].plot(target_volatilities, target_returns, label='Efficient Frontier', color='blue')
axes[0, 0].scatter(equal_portfolio_stats["Volatility"], equal_portfolio_stats["E(R)"], color='red', label='Equal Weight Portfolio')
axes[0, 0].scatter(optimized_portfolio_stats["Volatility"], optimized_portfolio_stats["E(R)"], color='green', label='Optimized Portfolio')
axes[0, 0].scatter(tangency_portfolio_stats["Volatility"], tangency_portfolio_stats["E(R)"], color='purple', label='Tangency Portfolio')
axes[0, 0].scatter(tangency_rf_portfolio_stats["Volatility"], tangency_rf_portfolio_stats["E(R)"], color='orange', label='Tangency Portfolio with RF')
# Get the coordinates of your two points
x1 = tangency_portfolio_stats["Volatility"]
y1 = tangency_portfolio_stats["E(R)"]
x2 = tangency_rf_portfolio_stats["Volatility"]
y2 = tangency_rf_portfolio_stats["E(R)"]

# Calculate the slope
slope = (y2 - y1) / (x2 - x1)

# Define how far you want to extend the line
# You can use the axis limits or set specific values
x_min = 0  # or axes[0, 0].get_xlim()[0]
x_max = axes[0, 0].get_xlim()[1]  # or a specific value like 0.5

# Calculate y values for the extended line using y = y1 + slope * (x - x1)
x_extended = np.array([x_min, x_max])
y_extended = y1 + slope * (x_extended - x1)

# Plot the extended line
axes[0, 0].plot(
    x_extended,
    y_extended,
    color='white',
    linestyle='--',
    linewidth=1,
    label='Capital Allocation Line'
)
axes[0, 0].set_xlabel('Volatility')
axes[0, 0].set_ylabel('Expected Return')
axes[0, 0].set_title('Efficient Frontier and Portfolios')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. Optimized portfolio weights bar chart
optimized_weights_bars = axes[0, 1].bar(optimized_weights.index, optimized_weights.values, color='cyan', edgecolor='black')
bar_chart_nums(optimized_weights_bars, axes[0, 1], optimized_weights.values)
axes[0, 1].set_title('Optimized Portfolio Weights')
axes[0, 1].set_ylabel('Weight')
axes[0, 1].set_xticklabels(optimized_weights.index, rotation=45)
axes[0, 1].grid(True)

# 3. Tangency portfolio weights bar chart
tangency_weights_bars = axes[1, 0].bar(tangency_portfolio_weights.index, tangency_portfolio_weights.values, color='magenta', edgecolor='black')
bar_chart_nums(tangency_weights_bars, axes[1, 0], tangency_portfolio_weights.values)
axes[1, 0].set_title('Tangency Portfolio Weights')
axes[1, 0].set_ylabel('Weight')
axes[1, 0].set_xticklabels(tangency_portfolio_weights.index, rotation=45)
axes[1, 0].grid(True)

# 4. Tangency portfolio with RF weights bar chart
tangency_rf_weights_bars = axes[1, 1].bar(tangency_rf_portfolio_weights.index, tangency_rf_portfolio_weights.values, color='yellow', edgecolor='black')
bar_chart_nums(tangency_rf_weights_bars, axes[1, 1], tangency_rf_portfolio_weights.values)
axes[1, 1].set_title('Tangency Portfolio with RF Weights')
axes[1, 1].set_ylabel('Weight')
axes[1, 1].set_xticklabels(tangency_rf_portfolio_weights.index, rotation=45)
axes[1, 1].grid(True)

plt.show()