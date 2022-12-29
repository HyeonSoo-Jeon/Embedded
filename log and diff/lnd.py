from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


series = pd.read_csv('./air/air.csv', header=0, index_col=0)  # 142개
series_log = log_series = np.log(series)
series_dif = series.diff(1).dropna()
series_dif1 = series.diff(1).diff(1).dropna()
series_dif2 = series.diff(1).diff(1).diff(1).dropna()

# plot_acf(series)
# plot_pacf(series)

# plot_acf(series_dif)
# plot_pacf(series_dif)

# plot_acf(series_dif1)
# plot_pacf(series_dif1)
# plt.show()


print(series.var(), '\n', series_dif.var())
print(adfuller(series))
print(adfuller(series_log))
print(adfuller(series_dif))
print(adfuller(series_dif1))
