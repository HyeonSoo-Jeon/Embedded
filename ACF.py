import statsmodels.api as sm

sm.graphics.tsa.plot_acf(result, lags=50, use_vlines=True)
