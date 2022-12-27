from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


series = pd.read_csv('./sales.csv', header=0, index_col=0)
# series['Sales_diff'] = series['Sales'].shift(periods=-1)
# series["Sales_diff"] = series['Sales'].diff(1)
# print(series)
# plot_acf(series['Sales_diff'])

plot_acf(series)
series.plot()


# plot_pacf(series)

# plt.show()

model = ARIMA(series[:30], order=(1, 2, 2))
model_fit = model.fit()
# print(model_fit.summary())
# print(model_fit.arparams)
plt.axvline(x=30, color='gray', linestyle='--')
model_fit.predict(end=40).plot()
plt.show()
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
