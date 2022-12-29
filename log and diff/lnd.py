from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


series = pd.read_csv('./air/air.csv', header=0, index_col=0)  # 142개
series_log1 = np.log(series)

series_dif1 = series.diff(1).dropna()
series_dif2 = series.diff(1).diff(1).dropna()
series_dif3 = series.diff(1).diff(1).diff(1).dropna()
# plot_acf(series)
# plot_pacf(series)

# plot_acf(series_dif1)
# plot_pacf(series_dif1)

# plot_acf(series_dif2)
# plot_pacf(series_dif2)

# plot_acf(series_dif3)
# plot_pacf(series_dif3)

# series.plot()
# series_dif1.plot()
# series_dif2.plot()
# series_dif3.plot()

# plt.show()

print(series.var(), '\n', series_dif1.var(), '\n', series_log1.var())
print(adfuller(series))
print(adfuller(series_log1))
print(adfuller(series_dif2))
print(adfuller(series_dif3))
