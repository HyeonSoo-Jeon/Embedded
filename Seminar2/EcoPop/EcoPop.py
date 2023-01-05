# 월별 경제활동 인구 (99.06 ~ 22.11)

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

series = pd.read_csv('./Seminar2/Ecopop/ecopop.csv',
                     parse_dates=['Date'],
                     header=0, index_col=0)  # 282개
# series.diff().dropna().plot()
# print(series.iloc[199])
print(series[:200].mean())
# 24406
model1 = ARIMA(series[:200], order=(0, 0, 1))
model_fit1 = model1.fit()
# print(model_fit1.summary())
# model2 = ARIMA(series[:200], order=(2, 0, 0))
# model_fit2 = model2.fit()
# model3 = ARIMA(series[:200], order=(2, 1, 0))
# model_fit3 = model3.fit()

series.plot()
model_fit1.predict(start=1, end=282).plot(label='0')
# model_fit2.predict(start=10, end=282).plot(label='1')
# model_fit3.predict(start=10, end=282).plot(label='2')
plt.axvline(x='2016-01-01', color='gray', linestyle='--')
plt.title('Monthly economically active population (99.06 ~ 22.11)')
plt.legend()
plt.show()

# print(model_fit1.predict(end=282)[199:202])
# print(model_fit2.predict(end=282)[199:203])
# print(model_fit3.predict(end=282)[199:204])
