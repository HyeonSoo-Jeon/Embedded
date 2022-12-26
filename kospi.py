from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


series = pd.read_csv('./kospi.csv', header=0, index_col=0)
# plot_acf(series)
# plot_pacf(series)
# plt.show()

model = ARIMA(series, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

print(model_fit.predict())
print(model_fit.forecast())
