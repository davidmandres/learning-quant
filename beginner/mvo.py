import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize

N = DAILY = 252
r_f = 0.01
r_prem = 0.06


EQUITY_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY"]
# RF_TICKER = "IEF"  # 7-10 year treasury ETF as a proxy for risk-free asset
TICKERS = EQUITY_TICKERS + ["RF"]

# mean-variance optimization (MVO) example
data = yf.download(tickers=EQUITY_TICKERS, period="1y", interval="1d")
prices = data['Close']

# calculate daily returns, annualized returns and volatilities
daily_returns = prices.pct_change().dropna()
daily_std = daily_returns.std()
annualized_returns = daily_returns.mean() * DAILY
annualized_std = daily_std * np.sqrt(DAILY)

# covariance stuff
cov_matrix = daily_returns.cov() * DAILY
cov_matrix_rf = cov_matrix.copy()
cov_matrix_rf["RF"] = 0.0
cov_matrix_rf.loc["RF"] = 0.0

# CAPM expected returns
ticker_betas = cov_matrix["SPY"] / cov_matrix.loc["SPY", "SPY"]
capm_returns = r_f + ticker_betas * r_prem

# simplified black-litterman expected returns, take the calculated (from the prices) expected returns and the capm returns, with equal weights
bl_expected_returns = 0.5 * (annualized_returns + capm_returns)
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

    bounds = tuple((0.1, 0.9) for _ in range(num_assets)) # no more than 90% in one asset, at least 10%

    # initial guess is equal weights
    initial_guess = np.array([1 / num_assets] * num_assets)
    
    result = minimize(neg, x0=initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    return pd.Series(result.x, index=assets)

equal_portfolio_weights = pd.Series(1 / len(EQUITY_TICKERS), index=EQUITY_TICKERS)
equal_portfolio_stats = portfolio_stats(equal_portfolio_weights)

print("Equal Portfolio Stats:")
print(equal_portfolio_stats)

optimized_weights = maximize_stat(portfolio_return, target_volatility=equal_portfolio_stats["Volatility"])
optimized_portfolio_stats = portfolio_stats(optimized_weights)

print("Optimized Portfolio Stats:")
print(optimized_portfolio_stats)

tangency_portfolio_weights = maximize_stat(sharpe_ratio)
tangency_portfolio_stats = portfolio_stats(tangency_portfolio_weights)

print("Tangency Portfolio Stats:")
print(tangency_portfolio_stats)

tangency_rf_portfolio_weights = maximize_stat(sharpe_ratio, assets=TICKERS, cov_matrix=cov_matrix_rf, bl_expected_returns=bl_expected_returns_rf)
tangency_rf_portfolio_stats = portfolio_stats(tangency_rf_portfolio_weights, bl_expected_returns=bl_expected_returns_rf, cov_matrix=cov_matrix_rf)

print("Tangency Portfolio with RF Stats:")
print(tangency_rf_portfolio_stats)

