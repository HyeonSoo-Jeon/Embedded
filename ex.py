from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()


data = pdr.get_data_yahoo("^KS11", start="2021-01-01", end="2022-01-01")

print(data)
