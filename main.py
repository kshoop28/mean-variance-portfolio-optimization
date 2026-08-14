import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize



# First two are Gold,
# Second two are Silver,
# Third two are Crude Oil
# Forth two are Agriculture
# Last two are Industrials

tickers = ['GLD','IAU','SLV','PPLT','USO','BNO','DBA','CORN','CPER','UNG']


# we can download historical prices by using the yahoo finance package
dfs = yf.download(tickers, start='2020-01-01', end='2026-01-01', auto_adjust=True)


prices = dfs['Close'][tickers]

# taking the previous days price and the current price to calc returns
ret = prices.pct_change().dropna()

# 252 trading days in a year

annret = ret.mean() * 252

covar = ret.cov() * 252

# we want to annualive both the returns and the covariance matrix

# We are now finding the global min variance portfolio

ones = np.ones(len(tickers))

covar_inverse = np.linalg.inv(covar.values)

gmv_weights = (
    covar_inverse @ ones
) / (
    ones.T @ covar_inverse @ ones
)

gmv_return = gmv_weights.T @ annret.values

gmv_variance = gmv_weights.T @ covar.values @ gmv_weights

gmv_volatility = np.sqrt(gmv_variance)


print("\nGLOBAL MINIMUM VARIANCE PORTFOLIO")
for ticker, weight in zip(tickers, gmv_weights):
    print(f"{ticker}: {weight:.4f}")

print(f"\nExpected Return: {gmv_return:.4f}")
print(f"Variance: {gmv_variance:.4f}")
print(f"Volatility: {gmv_volatility:.4f}")
print(f"Sum of Weights: {gmv_weights.sum():.4f}")


# in python matrix multiplication is denoted using @

def portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights


# Long-Only Global Minimum Variance Portfolio


initial_weights = np.ones(len(tickers)) / len(tickers)

constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

bounds = [(0, 1)] * len(tickers)

lgmvres = minimize(portfolio_variance, initial_weights, args=(covar.values,),method='SLSQP',bounds=bounds,constraints=constraints)

lgmvweight = lgmvres.x

lgmvret = lgmvweight @ annret.values

lgmvar = (lgmvweight.T @ covar.values @ lgmvweight)

lgmvol = np.sqrt(lgmvar)


print("\nLONG-ONLY GLOBAL MINIMUM VARIANCE PORTFOLIO")

for ticker, weight in zip(tickers, lgmvweight):
    print(f"{ticker}: {weight:.4f}")

print(f"\nExpected Return: {lgmvret:.4f}")
print(f"Variance: {lgmvar:.4f}")
print(f"Volatility: {lgmvol:.4f}")
print(f"Sum of Weights: {lgmvweight.sum():.4f}")


# Function to find minimum-risk portfolio for a given target return


def optimize_portfolio(target_return):

    initial_weights = np.ones(len(tickers)) / len(tickers)

    constraints = [{
            'type': 'eq',
            'fun': lambda weights: np.sum(weights) - 1
        },
                 
        {
            'type': 'eq',
            'fun': lambda weights:
                weights @ annret.values - target_return
        }
    ]

    # No short selling
    bounds = [(0, 1)] * len(tickers)

    result = minimize(portfolio_variance, initial_weights, args=(covar.values,), method='SLSQP', bounds=bounds, constraints=constraints)

    return result


# Generate the Efficient Frontier


minimum_return = annret.min()
maximum_return = annret.max()

target_returns = np.linspace(minimum_return,maximum_return,100)

frontier_returns = []
frontier_volatilities = []
frontier_weights = []


for target_return in target_returns:

    result = optimize_portfolio(target_return)

    if result.success:

        weights = result.x
        portfolio_return = (weights @ annret.values)

        portfolio_variance_value = (weights.T @ covar.values @ weights)
        portfolio_volatility = np.sqrt(portfolio_variance_value)

        # The paper only uses the efficient upper branch.
        if portfolio_return >= lgmvret:
            frontier_returns.append(portfolio_return)

            frontier_volatilities.append(portfolio_volatility)

            frontier_weights.append(weights)


# The sharpe ratio is the number that quantifies risk adjusted returns
# Find Maximum Sharpe Ratio Portfolio

risk_free_rate = 0.04


def negative_sharpe(weights):
    portfolio_return = (weights @ annret.values)
    portfolio_variance_value = (weights.T @ covar.values @ weights)
    portfolio_volatility = np.sqrt(portfolio_variance_value)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio


sharpe_result = minimize(negative_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)


max_sharpe_weights = sharpe_result.x

max_sharpe_return = (max_sharpe_weights @ annret.values)

max_sharpe_variance = (max_sharpe_weights.T @ covar.values @ max_sharpe_weights)

max_sharpe_volatility = np.sqrt( max_sharpe_variance)

max_sharpe_ratio = (max_sharpe_return - risk_free_rate) / max_sharpe_volatility


print("\nMAXIMUM SHARPE RATIO PORTFOLIO")

for ticker, weight in zip(tickers, max_sharpe_weights):
    print(f"{ticker}: {weight:.4f}")

print(f"\nExpected Return: " f"{max_sharpe_return:.4f}")

print(f"Volatility: " f"{max_sharpe_volatility:.4f}")

print(f"Sharpe Ratio: " f"{max_sharpe_ratio:.4f}")


# plotting the efficient frontier

plt.figure(figsize=(10, 6))

plt.plot(frontier_volatilities, frontier_returns, label='Efficient Frontier')

plt.scatter(lgmvol, lgmvret, marker='o', s=100, label='Long-Only GMV')

plt.scatter(max_sharpe_volatility, max_sharpe_return, marker='*', s=200, label='Maximum Sharpe Ratio')

plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Expected Return')
plt.title('Markowitz Mean-Variance Efficient Frontier')

plt.legend()
plt.grid()


# plotting the correlation matrix

correlation = ret.corr()

plt.figure(figsize=(8, 6))
plt.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.xticks(range(len(tickers)), tickers, rotation=45)
plt.yticks(range(len(tickers)), tickers)
plt.title('Commodity ETF Correlation Matrix')
plt.tight_layout()
plt.show()
