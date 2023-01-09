# 이산화 질소 월별 대기 오염도

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pmdarima import auto_arima


series = pd.read_csv('./Seminar1/population/uspop.csv',
                     header=0, index_col=0)  # 142개

# opt = auto_arima(series, seasonal=False, trace=True,
#                  information_criterion='bic')
model1 = ARIMA(series[:120], order=(2, 0, 0))
model_fit1 = model1.fit()
model2 = ARIMA(series[:120], order=(0, 0, 2))
model_fit2 = model2.fit()
model3 = ARIMA(series[:120], order=(2, 1, 0))
model_fit3 = model3.fit()


series.plot()
model_fit1.predict(start=5, end=142).plot(label="AR(2) Predictions")
model_fit2.predict(start=5, end=142).plot(label="MA(2) Predictions")
model_fit3.predict(start=5, end=142).plot(label="ARIMA(2, 1, 0)")
plt.legend()
plt.show()

# minAIC = float('inf')
# aicPDQ = -1
# minBIC = float('inf')
# bicPDQ = -1
# minHI = float('inf')
# HIPDQ = -1

# for i in range(1, 10):
#     model = ARIMA(series, order=(0, 0, i))
#     model_fit = model.fit()
#     aic = model_fit.aic
#     bic = model_fit.bic
#     hi = model_fit.hqic

#     if minAIC > aic:
#         minAIC = aic
#         aicPDQ = i

#     if minBIC > bic:
#         minBIC = bic
#         bicPDQ = i

#     if minHI > hi:
#         minHI = hi
#         HIPDQ = i

# print(
#     f"AIC : {minAIC}, {aicPDQ}\nBIC : {minBIC}, {bicPDQ}\nHQIC : {minHI}, {HIPDQ}")
