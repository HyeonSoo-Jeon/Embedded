from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


series = pd.read_csv('./KOSPI/kospi.csv', header=0, index_col=0)


# plot_acf(series)
# plot_pacf(series)
# plt.show()

series[:300].plot()

# model = ARIMA(series, order=(0, 1, 0))
model = ARIMA(series[:100], order=(0, 2, 1))
model_fit = model.fit()
print(model_fit.summary())

print('\n[predict & forecast]')
# model_fit.predict(end=300)[10:].plot()
model_fit.forecast(200).plot()
# print(model_fit.predict(end=200))
plt.axvline(x=100, color='r', linestyle='--')
plt.show()

result = adfuller(model_fit.predict())
print(result)

# 12/23/2022	2317.26
# (1,1,1)       2356.417042
# (2,1,2)       2356.63802
