# 이산화 질소 월별 대기 오염도

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

series = pd.read_csv('./air/air.csv', header=0, index_col=0)  # 142개


model = ARIMA(series[:100], order=(2, 1, 2))
model_fit = model.fit()
print(series)
print(model_fit.predict())

series.plot()
plt.axvline(x=100, color='gray', linestyle='--')
model_fit.predict(end=142).plot(label='predict')
plt.legend()
plt.show()

print(model_fit.summary())

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
