import pandas as pd
import matplotlib.pyplot as plt


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

# 1
# plt.subplot(241)
fig, ax1 = plt.subplots()
plt.xticks(rotation=45)
ax1.scatter(df1['ds'], df1['y'], color='green', label='S', s=10)
ax2 = ax1.twinx()
ax2.scatter(df1['ds'], df1['H'], color='blue', label='H', s=10)
# plt.legend(['S', 'H'], loc="best")
plt.axvline(x=pd.datetime(2021, 11, 3), color='r', linestyle='--', linewidth=1)
plt.title('H-S_seq_1')
plt.xlabel('ds')
# 2
# plt.subplot(242)
fig, ax1 = plt.subplots()
plt.xticks(rotation=45)
ax1.scatter(df2['ds'], df2['y'], color='green', label='S', s=10)
ax2 = ax1.twinx()
ax2.scatter(df2['ds'], df2['H'], color='blue', label='H', s=10)
# plt.legend(['S', 'H'], loc="best")
plt.axvline(x=pd.datetime(2021, 11, 3), color='r', linestyle='--', linewidth=1)
plt.title('H-S_seq_2')
plt.xlabel('ds')
# 3
# plt.subplot(243)
fig, ax1 = plt.subplots()
plt.xticks(rotation=45)
ax1.scatter(df3['ds'], df3['y'], color='green', label='S', s=10)
ax2 = ax1.twinx()
ax2.scatter(df3['ds'], df3['H'], color='blue', label='H', s=10)
# plt.legend(['S', 'H'], loc="best")
plt.axvline(x=pd.datetime(2021, 11, 3), color='r', linestyle='--', linewidth=1)
plt.title('H-S_seq_3')
plt.xlabel('ds')
# 4
# plt.subplot(244)
fig, ax1 = plt.subplots()
plt.xticks(rotation=45)
ax1.scatter(df3['ds'], df3['y'], color='green', label='S', s=10)
ax2 = ax1.twinx()
ax2.scatter(df3['ds'], df3['H'], color='blue', label='H', s=10)
# plt.legend(['S', 'H'], loc="best")
plt.axvline(x=pd.datetime(2021, 11, 3), color='r', linestyle='--', linewidth=1)
plt.title('H-S_seq_4')
plt.xlabel('ds')
# 5
# plt.subplot(244)
fig, ax1 = plt.subplots()
plt.xticks(rotation=45)
ax1.scatter(df5['ds'], df5['y'], color='green', label='S', s=10)
ax2 = ax1.twinx()
ax2.scatter(df5['ds'], df5['H'], color='blue', label='H', s=10)
# plt.legend(['S', 'H'], loc="best")
plt.axvline(x=pd.datetime(2021, 11, 3), color='r', linestyle='--', linewidth=1)
plt.title('H-S_seq_5')
plt.xlabel('ds')
# 6
# plt.subplot(244)
fig, ax1 = plt.subplots()
plt.xticks(rotation=45)
ax1.scatter(df6['ds'], df6['y'], color='green', label='S', s=10)
ax2 = ax1.twinx()
ax2.scatter(df6['ds'], df6['H'], color='blue', label='H', s=10)
# plt.legend(['S', 'H'], loc="best")
plt.axvline(x=pd.datetime(2021, 11, 3), color='r', linestyle='--', linewidth=1)
plt.title('H-S_seq_6')
plt.xlabel('ds')
# 7
# plt.subplot(244)
fig, ax1 = plt.subplots()
plt.xticks(rotation=45)
ax1.scatter(df7['ds'], df7['y'], color='green', label='S', s=10)
ax2 = ax1.twinx()
ax2.scatter(df7['ds'], df7['H'], color='blue', label='H', s=10)
# plt.legend(['S', 'H'], loc="best")
plt.axvline(x=pd.datetime(2021, 11, 3), color='r', linestyle='--', linewidth=1)
plt.title('H-S_seq_7')
plt.xlabel('ds')
# 8
# plt.subplot(244)
fig, ax1 = plt.subplots()
plt.xticks(rotation=45)
ax1.scatter(df8['ds'], df8['y'], color='green', label='S', s=10)
ax2 = ax1.twinx()
ax2.scatter(df8['ds'], df8['H'], color='blue', label='H', s=10)
# plt.legend(['S', 'H'], loc="best")
plt.axvline(x=pd.datetime(2021, 11, 3), color='r', linestyle='--', linewidth=1)
plt.title('H-S_seq_8')
plt.xlabel('ds')

plt.show()
