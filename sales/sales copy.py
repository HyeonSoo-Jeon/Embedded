from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np


series = pd.read_csv('./sales.csv', header=0, index_col=0)

print(adfuller(series, autolag='AIC'))

log_series = np.log(series)
print(adfuller(log_series, autolag='AIC'))

df_series = log_series.diff(1).diff(1)
# print(df_series)
print(adfuller(df_series[2:]))
df_series.plot()


# df_series.plot()

# model = ARIMA(series, order=(0, 1, 0))
model = ARIMA(log_series, order=(1, 2, 1))
model_fit = model.fit()
print(adfuller(model_fit.predict()))
model_fit.predict().plot()
# plt.show()

# plot_pacf(series)

# plt.show()
# print(model_fit.)

# print('\n[predict & forecast]')
# print(model_fit.predict(end=40))
# print(model_fit.forecast(steps=10))

# with open('sales.txt', 'w') as f:
#     model = ARIMA(series, order=(1, 0, 0))
#     model_fit = model.fit()
#     f.write(str(model_fit.summary()))
#     f.write(str('\n[predict & forecast]'))
#     f.write(str(model_fit.predict()))
#     f.write(str(model_fit.forecast(steps=10)))
#     f.write('\n')

#     model = ARIMA(series, order=(2, 0, 0))
#     model_fit = model.fit()
#     f.write(str(model_fit.summary()))
#     f.write(str('\n[predict & forecast]'))
#     f.write(str(model_fit.predict()))
#     f.write(str(model_fit.forecast(steps=10)))
#     f.write('\n')

#     model = ARIMA(series, order=(0, 0, 1))
#     model_fit = model.fit()
#     f.write(str(model_fit.summary()))
#     f.write(str('\n[predict & forecast]'))
#     f.write(str(model_fit.predict()))
#     f.write(str(model_fit.forecast(steps=10)))
#     f.write('\n')

#     model = ARIMA(series, order=(0, 0, 2))
#     model_fit = model.fit()
#     f.write(str(model_fit.summary()))
#     f.write(str('\n[predict & forecast]'))
#     f.write(str(model_fit.predict()))
#     f.write(str(model_fit.forecast(steps=10)))
#     f.write('\n')


# 12/23/2022	2317.26
# (1,1,1)       2356.417042
# (2,1,2)       2356.63802
