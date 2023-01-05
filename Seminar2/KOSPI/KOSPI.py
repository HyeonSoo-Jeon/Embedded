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

# result = seasonal_decompose(series, model='add')
# result.plot()
# plt.show()
