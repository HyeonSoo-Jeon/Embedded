# 월별 경제활동 인구 (99.06 ~ 22.11)

from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")


series = pd.read_csv('./Seminar2/Ecopop/ecopop.csv',
                     parse_dates=['Date'],
                     header=0, index_col=0)

auto_arima(series[:200], seasonal=False,
           trace=True, information_criterion='bic')
# model = ARIMA(series[:200], order=(2, 1, 3))
# series.plot()

# model_fit = model.fit()
# model_fit.predict(start=2, end=282).plot(label='ARIMA(2, 1, 3)')

# plt.axvline(x='2016-01-01', color='gray', linestyle='--')
# plt.title('Monthly economically active population (99.06 ~ 22.11)')
# plt.legend()
# plt.show()
