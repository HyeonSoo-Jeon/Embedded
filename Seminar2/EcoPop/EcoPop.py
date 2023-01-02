# 경제활동 인구 99.06 ~ 22.11

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

series = pd.read_csv('./Seminar2/Ecopop/ecopop.csv',
                     header=0, index_col=0)  # 282개
ax = plt.axes()
print(len(series))
series.plot()
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
plt.show()
