# 경제활동 인구 99.06 ~ 22.11

from statsmodels.tsa.seasonal import seasonal_decompose      # for ETS Plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")


series = pd.read_csv('./Seminar2/KOSPI/KOSPI.csv',
                     header=0, index_col=0)  # 282개

# opt = auto_arima(series[:11000], seasonal=False,
#                  trace=True, d=2, information_criterion='bic')
model = ARIMA(series[:11000], order=(0, 2, 2))
model_fit = model.fit()

series.plot()
model_fit.predict(start=2, end=12965).plot(label='ARIMA(0, 2, 2)')

plt.axvline(x=11000, color='gray', linestyle='--')
plt.title('Daily KOSPI closing price (75.01.04 ~ 23.01.04)')
plt.xticks(rotation=45)
plt.legend()
plt.show()
