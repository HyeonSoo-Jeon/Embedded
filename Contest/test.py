import pandas as pd
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet


def make_df(filename, ID):
    df = pd.read_csv(filename)
    df.rename(columns={'Date': 'ds', 'S': 'y'}, inplace=True)

    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index('ds', inplace=True)
    df_daily = df.resample('D').mean()

    df_daily['H'].fillna(method='ffill', inplace=True)
    df = df_daily.reset_index()
    df['ID'] = ID

    df = df[['ds', 'y', 'H', 'ID']]

    return df


def make_df_de(filename, ID):
    df = pd.read_csv(filename)
    df.rename(columns={'Date': 'ds', 'S': 'y'}, inplace=True)

    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index('ds', inplace=True)
    df_daily = df.resample('D').mean()

    df = df_daily.reset_index()
    df['ID'] = ID

    df = df[['ds', 'y', 'H', 'ID']]

    return df


df1 = make_df_de('./contest/H-S_seq_1.csv', ID='d')
df2 = make_df('./contest/H-S_seq_1.csv', ID='data1')

plt.scatter(df1['ds'], df1['H'], label='H', s=10)
plt.title('H-S_seq_1')
plt.legend(loc="best")
plt.xlabel('Date')
plt.ylabel('H')
plt.xticks(rotation=45)
plt.locator_params(axis='x', nbins=8)
plt.show()


plt.scatter(df2['ds'], df2['H'], label='H', s=10)
plt.title('H-S_seq_1')
plt.legend(loc="best")
plt.xlabel('Date')
plt.ylabel('H')
plt.xticks(rotation=45)
plt.show()
