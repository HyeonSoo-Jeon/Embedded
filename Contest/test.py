import pandas as pd
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet


def make_df(filename, ID, slicing=0):
    df = pd.read_csv(f'./contest/{filename}')

    if slicing:
        l = len(df)
        df = df.head(slicing)
        print(f'sliced into {len(df)} out of {l} data.')

    df.rename(columns={'Date': 'ds', 'S': 'y'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index('ds', inplace=True)
    df_daily = df.resample('D').mean()
    df = df_daily.reset_index()

    df_daily['H'].fillna(method='ffill', inplace=True)

    df_daily = df_daily.interpolate(method='polynomial', order=2)

    df['ID'] = ID
    df = df[['ds', 'y', 'H', 'ID']]
    # df.set_index('ds', inplace=True)

    return df


df1 = make_df('H-S_seq_1.csv', ID='data1')
df2 = make_df('H-S_seq_2.csv', ID='data2')
df3 = make_df('H-S_seq_3.csv', ID='data3')
df4 = make_df('H-S_seq_4.csv', ID='data4')
df5 = make_df('H-S_seq_5.csv', ID='data5')
df6 = make_df('H-S_seq_6.csv', ID='data6')
df7 = make_df('H-S_seq_7.csv', ID='data7')
df8 = make_df('H-S_seq_8.csv', ID='data8')

pm = NeuralProphet(learning_rate=0.1,
                   yearly_seasonality=False,
                   weekly_seasonality=False,
                   unknown_data_normalization=True,
                   newer_samples_weight=4,
                   n_forecasts=10,
                   drop_missing=True,
                   optimizer='AdamW',
                   )

pm = pm.add_future_regressor(name='H', normalize=True)
metrics = pm.fit(df1, progress='plot')
