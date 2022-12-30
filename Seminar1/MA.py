from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


series = pd.read_csv('./test.csv', header=0, index_col=0)

# plot_acf(series)
# plot_pacf(series)
# plt.show()

# series.plot()

model = ARIMA(series, order=(0, 0, 1))
# model = ARIMA(series, order=(2, 1, 2))
model_fit = model.fit()
print(model_fit.summary())

print('\n[predict]')
print(model_fit.predict())

print('\n[forecast]')
print(model_fit.forecast(steps=10))

# 12/23/2022	2317.26
# (1,1,1)       2356.417042
# (2,1,2)       2356.63802
