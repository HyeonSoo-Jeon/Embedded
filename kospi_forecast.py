from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


series = pd.read_csv('./kospi.csv', header=0, index_col=0)


# plot_acf(series)
# plot_pacf(series)
# plt.show()

series.plot()

# model = ARIMA(series, order=(0, 1, 0))
model = ARIMA(series[:400], order=(1, 1, 2))
model_fit = model.fit()
print(model_fit.summary())

print('\n[predict & forecast]')
model_fit.predict(end=750)[100:].plot()
print(model_fit.predict(end=750))
plt.axvline(x=400, color='r', linestyle='--')
plt.show()


# 12/23/2022	2317.26
# (1,1,1)       2356.417042
# (2,1,2)       2356.63802
