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

opt = auto_arima(series[:125], seasonal=False,
                 trace=True, information_criterion='bic')
# print(opt.summary())
# # 내생각 (1,1,1)(1,1,1,12)
# # best (2,1,2)(2,0,1,12)
# model1 = ARIMA(series[:125], order=(0, 1, 0))
# model_fit1 = model1.fit()
# series.plot()
# plt.axvline(x=125, color='gray', linestyle='--')
# model_fit1.predict(start=2, end=142).plot(label='ARIMA(0, 1, 0)')
# plt.legend()
# plt.show()

# mse1 = mean_squared_error(series[:100], model_fit1.predict())
# mse2 = mean_squared_error(series[:100], model_fit2.predict())
# print(mse1)
# print(mse2)

# print(model_fit.summary())

# print(model_fit.summary())


# model2 = ARIMA(series[:100], order=(1, 2, 1), seasonal_order=(1, 1, 1, 2))
# model2_fit = model2.fit()
# model2_fit.predict(end=142).plot()

# minAIC = float('inf')
# pdq = [0, 0, 0]

# with open('air.txt', 'w') as f:
#     for p in range(5):
#         for d in range(1, 4):
#             for q in range(5):
#                 AIC = ARIMA(series[:100], order=(p, d, q)).fit().aic
#                 f.write(f'[{p}, {d}, {q}] : {AIC}\n')
#                 if abs(AIC) < minAIC:
#                     pdq = [p, d, q]

# print(pdq)
