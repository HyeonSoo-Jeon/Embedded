import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm


series = pd.read_csv('./sales.csv', header=0, index_col=0)
# plot_acf(series)
# plot_pacf(series)
# plt.show()
# model = ARIMA(series, order=(p, d, q))
# model_fit = model.fit()
# print(model_fit.summary())
model = ARIMA(series, order=(2, 2, 2))
model_fit = model.fit()
# print(model_fit.summary())
# fore = model_fit.get_forecast()
print(model_fit.predict())
with open('arima.txt', 'w') as f:
    for p in range(20):
        for d in range(1, 3):
            for q in range(20):
                model = ARIMA(series, order=(p, d, q))
                model_fit = model.fit()
                f.write(str(model_fit.summary()))
