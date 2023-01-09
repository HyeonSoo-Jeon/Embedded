# 이산화 질소 월별 대기 오염도

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

series = pd.read_csv('./Seminar1/air/air.csv', header=0, index_col=0)  # 142개

series.plot()
plt.show()
# plot_acf(series[:100])
# plot_pacf(series[:100])

# series_diff1 = series[:100].diff().dropna()
# series_diff2 = series[:100].diff().diff().dropna()
# series_diff1.plot()
# series_diff2.plot()

# model2 = ARIMA(series[:125], order=(0, 1, 0))
# model_fit2 = model2.fit()

# series.plot()
# plt.axvline(x=125, color='gray', linestyle='--')
# model_fit2.predict(end=142).plot(label='predict')
# plt.legend()
# plt.show()


# opt = auto_arima(series[:125], seasonal=False, d=2,
#                  trace=True, information_criterion='bic')
# print(opt.summary())
