import numpy as np
import pandas as pd
import yfinance as yf

'''Gold = ['GLD','IAU'] 
   Silver / Precious Metals = ['SLV','PPLT']
   Crude Oil = ['USO', 'SCO'] 
   Agriculture = ['DBA','CORN']
   Industrial / Energies = ['CPER','UNG']
   

'''


tickers = ['GLD','IAU','SLV','PPLT','USO','BNO','DBA','CORN','CPER','UNG']

dfs = yf.download(tickers, start = '2020-01-01', end = '2026-01-01')

ret = (dfs['Close'] / dfs['Close'].shift() - 1).dropna()

annret = ret.mean() * 252 # We use 252 because there is only 252 trading days in a year counting for holidays and weekends
print("Expected Returns")
print(annret)

covar = ret.cov() * 252

print("\nCovariance Matrix")
print(covar)


# Now lets create our weights

# The weight is going to be a vector of individual weights,
ones = np.ones(10)

inverse_array = np.linalg.inv(covar.values)

# 3. Convert the result back into a Pandas DataFrame
covar_inverse = pd.DataFrame(inverse_array, index=covar.columns, columns=covar.index)
print("\nInverse DataFrame:")
print(covar_inverse)


w = (covar_inverse @ ones) / (ones.transpose() @ covar_inverse @ ones)


print(w)
# To verify that this is the inverse
"""
identity_matrix = covar.dot(covar_inverse)
print("\nVerification (Identity Matrix):")
print(identity_matrix.round(2))
"""