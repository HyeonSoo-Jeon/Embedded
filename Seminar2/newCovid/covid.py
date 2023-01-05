# 일간 새로운 확진자
from statsmodels.tsa.seasonal import seasonal_decompose      # for ETS Plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

series = pd.read_csv('./Seminar2/newCovid/COVID19.csv',
                     parse_dates=['Date'],
                     header=0, index_col=0)  # 307개

result = seasonal_decompose(series, model='add')
result.plot()
plt.show()
