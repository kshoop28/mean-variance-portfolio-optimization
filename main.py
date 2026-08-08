import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# --------------------------------------------------
'''Gold = ['GLD','IAU'] 
   Silver / Precious Metals = ['SLV','PPLT']
   Crude Oil = ['USO', 'SCO'] 
   Agriculture = ['DBA','CORN']
   Industrial / Energies = ['CPER','UNG']
'''
# --------------------------------------------------

tickers = ['GLD','IAU','SLV','PPLT','USO','BNO','DBA','CORN','CPER','UNG']


# --------------------------------------------------
# 2. Download historical prices
# --------------------------------------------------

dfs = yf.download(tickers, start = '2020-01-01', end = '2026-01-01')


prices = dfs['Close']


# --------------------------------------------------
# 3. Calculate daily returns
# --------------------------------------------------

ret = prices.pct_change().dropna()


# --------------------------------------------------
# 4. Calculate expected annual returns
# --------------------------------------------------

# 252 trading days in a year

annret = ret.mean() * 252

# --------------------------------------------------
# 5. Calculate annualized covariance matrix
# --------------------------------------------------

covar = ret.cov() * 252

# --------------------------------------------------
# 6. Global Minimum Variance Portfolio
# --------------------------------------------------

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
print("-----------------------------------")

for ticker, weight in zip(tickers, gmv_weights):
    print(f"{ticker}: {weight:.4f}")

print(f"\nExpected Return: {gmv_return:.4f}")
print(f"Variance: {gmv_variance:.4f}")
print(f"Volatility: {gmv_volatility:.4f}")
print(f"Sum of Weights: {gmv_weights.sum():.4f}")


# --------------------------------------------------
# 7. Function to calculate portfolio variance
# --------------------------------------------------

def portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights


# --------------------------------------------------
# 8. Function to find minimum-risk portfolio
#    for a given target return
# --------------------------------------------------

def optimize_portfolio(target_return):

    initial_weights = np.ones(len(tickers)) / len(tickers)

    constraints = [
        {
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

    result = minimize(
        portfolio_variance,
        initial_weights,
        args=(covar.values,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result


# --------------------------------------------------
# 9. Generate the Efficient Frontier
# --------------------------------------------------

minimum_return = annret.min()
maximum_return = annret.max()

target_returns = np.linspace(
    minimum_return,
    maximum_return,
    100
)

frontier_returns = []
frontier_volatilities = []
frontier_weights = []


for target_return in target_returns:

    result = optimize_portfolio(target_return)

    if result.success:

        weights = result.x

        portfolio_return = (
            weights @ annret.values
        )

        portfolio_variance_value = (
            weights.T @ covar.values @ weights
        )

        portfolio_volatility = np.sqrt(
            portfolio_variance_value
        )

        frontier_returns.append(portfolio_return)

        frontier_volatilities.append(
            portfolio_volatility
        )

        frontier_weights.append(weights)


# --------------------------------------------------
# 10. Find Maximum Sharpe Ratio Portfolio
# --------------------------------------------------

# You can change this later to the risk-free rate
# you want to use in the paper.
risk_free_rate = 0.04


def negative_sharpe(weights):

    portfolio_return = (
        weights @ annret.values
    )

    portfolio_variance_value = (
        weights.T @ covar.values @ weights
    )

    portfolio_volatility = np.sqrt(
        portfolio_variance_value
    )

    sharpe_ratio = (
        portfolio_return - risk_free_rate
    ) / portfolio_volatility

    return -sharpe_ratio


constraints = {
    'type': 'eq',
    'fun': lambda weights: np.sum(weights) - 1
}

bounds = [(0, 1)] * len(tickers)

initial_weights = np.ones(len(tickers)) / len(tickers)


sharpe_result = minimize(
    negative_sharpe,
    initial_weights,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)


max_sharpe_weights = sharpe_result.x

max_sharpe_return = (
    max_sharpe_weights @ annret.values
)

max_sharpe_variance = (
    max_sharpe_weights.T
    @ covar.values
    @ max_sharpe_weights
)

max_sharpe_volatility = np.sqrt(
    max_sharpe_variance
)

max_sharpe_ratio = (
    max_sharpe_return - risk_free_rate
) / max_sharpe_volatility


print("\nMAXIMUM SHARPE RATIO PORTFOLIO")
print("-----------------------------------")

for ticker, weight in zip(
    tickers,
    max_sharpe_weights
):
    print(f"{ticker}: {weight:.4f}")

print(
    f"\nExpected Return: "
    f"{max_sharpe_return:.4f}"
)

print(
    f"Volatility: "
    f"{max_sharpe_volatility:.4f}"
)

print(
    f"Sharpe Ratio: "
    f"{max_sharpe_ratio:.4f}"
)


# --------------------------------------------------
# 11. Plot Efficient Frontier
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    frontier_volatilities,
    frontier_returns,
    label='Efficient Frontier'
)

plt.scatter(
    gmv_volatility,
    gmv_return,
    marker='o',
    s=100,
    label='Global Minimum Variance'
)

plt.scatter(
    max_sharpe_volatility,
    max_sharpe_return,
    marker='*',
    s=200,
    label='Maximum Sharpe Ratio'
)

plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Expected Return')
plt.title(
    'Markowitz Mean-Variance Efficient Frontier'
)

plt.legend()
plt.grid()

plt.show()