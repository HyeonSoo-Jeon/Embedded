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
import warnings
warnings.filterwarnings("ignore")


series = pd.read_csv('./Seminar2/KOSPI/KOSPI.csv',
                     header=0, index_col=0)  # 282개


print(series[:10000].mean())

# print(series.rolling(2).mean().mean())
# print(series.rolling(3).mean().mean())
model1 = ARIMA(series[:10000], order=(0, 0, 0))
model_fit1 = model1.fit()
print(model_fit1.predict())
# model2 = ARIMA(series[:10000], order=(0, 0, 1))
# model_fit2 = model2.fit()
# model3 = ARIMA(series[:10000], order=(0, 0, 2))
# model_fit3 = model3.fit()
# print(model1.param_names)
# print(model2.param_names)
# print(model3.param_names)
# print(model3.state_names)

# series.plot()
# model_fit1.predict(end=12965).plot(label='0')
# model_fit2.predict(end=12965).plot(label='1')
# model_fit3.predict(end=12965).plot(label='2')
# plt.axvline(x=10000, color='gray', linestyle='--')
# plt.legend()
# plt.show()
