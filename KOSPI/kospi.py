from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np


series = pd.read_csv('./KOSPI/kospi.csv', header=0, index_col=0)

plot_acf(series)
plot_pacf(series)
# series_dif1 = series.diff().dropna()
# series_dif2 = series.diff().diff().dropna()

# series_dif1.plot()
# series_dif2.plot()

plt.show()
# series.plot()
# model = ARIMA(series[:600], order=(1, 2, 0))
# model_fit = model.fit()

# model_fit.predict(end=721).plot()
# plt.axvline(x=600, color='gray', linestyle='--')
# plt.show()
# print(model_fit.summary())

# print(model_fit.predict())
# print(model_fit.forecast())

# 12/23/2022	2317.26
# (1,1,1)       2356.417042
# (2,1,2)       2356.63802

# with open('kospi.txt', 'w') as f:
#     for p in range(5):
#         for d in range(1, 3):
#             for q in range(5):
#                 model = ARIMA(series[:100], order=(p, d, q))
#                 model_fit = model.fit()
#                 out = f'({p}, {d}, {q}) : AIC - {model_fit.aic}, BIC - {model_fit.bic}'
#                 print(out)
#                 f.write(out)
#                 f.write('\n')
