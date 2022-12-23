import pandas as pd

series = pd.read_csv('sales.csv', header=0, index_col=0, squeeze=True)
series.plot()
