from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from pmdarima import auto_arima


series = pd.read_csv('./KOSPI/kospi.csv', header=0, index_col=0)

# opt = auto_arima(series, seasonal=False, trace=True, d=1,
#                  information_criterion='bic')
# print(opt.summary())
# opt = auto_arima(series, seasonal=False, trace=True, d=2,
#                  information_criterion='bic')
# print(opt.summary())

series.plot()

model = ARIMA(series[:600], order=(0, 1, 0))
model_fit = model.fit()

model_fit.predict(end=721).plot(label='Predictions')
plt.axvline(x=600, color='gray', linestyle='--')

plt.show()
