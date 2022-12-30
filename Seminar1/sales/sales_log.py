from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

series = pd.read_csv('./sales/sales.csv', header=0, index_col=0)
series_log1 = np.log(series)

series.plot()
series_log1.plot()

plt.show()
