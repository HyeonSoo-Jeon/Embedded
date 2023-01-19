import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv('./Contest/EX/DHS_Daily_Report_2022.csv')
print(df)
