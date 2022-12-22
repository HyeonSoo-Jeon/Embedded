import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import yfinance as yf
from statsmodels.tsa.arima_model import ARIMA
import datetime

plt.style.use('dark_background')

series = yf.download('^KS11', start="2021-01-01", end="2022-01-01").Close
model = ARIMA(series, order=(0, 1, 1))
model_fit = model.fit(trend='nc', full_output=True, disp=1)
print(model_fit.summary())
