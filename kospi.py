from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


series = pd.read_csv('./kospi.csv', header=0, index_col=0)


# model = ARIMA(series, order=(0, 1, 0))
# model = ARIMA(series, order=(2, 1, 2))
# model_fit = model.fit()
# print(model_fit.summary())

# print(model_fit.predict())
# print(model_fit.forecast())

# 12/23/2022	2317.26
# (1,1,1)       2356.417042
# (2,1,2)       2356.63802

with open('kospi.txt', 'w') as f:
    for p in range(10):
        for d in range(1, 3):
            for q in range(10):
                model = ARIMA(series, order=(p, d, q))
                model_fit = model.fit()
                out = f'({p}, {d}, {q}) : AIC - {model_fit.aic}, BIC - {model_fit.bic})'
                print(out)
                f.write(out)
                f.write('\n')
