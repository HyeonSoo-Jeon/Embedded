from ant import *

from statsmodels.tsa.arima.model import ARIMA


model = ARIMA(series[:200], order=(1, 1, 1))
model_fit = model.fit()
model_fit.predict(start=1, end=282).plot(label='ARIMA(1, 0, 0)')
