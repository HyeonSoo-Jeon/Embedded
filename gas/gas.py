from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


series = pd.read_csv('./gas/gas.csv', header=0, index_col=0)

series.plot()
plot_acf(series)
plot_pacf(series)
plt.show()

# print(adfuller(series))
