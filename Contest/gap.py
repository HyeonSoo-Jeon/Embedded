import pandas as pd
import matplotlib.pyplot as plt


def gap(df):
    l = 0
    s = 0
    # before = df.iloc[18]['S']
    # for idx, row in df[19:].iterrows():
    before = df.iloc[21]['S']
    # for idx, row in df[22:30].iterrows():
    for idx, row in df[22:30].iterrows():
        g = (row['S']-before)**2
        # if g >= 1:
        s += g
        l += 1
        before = row['S']
    return s/l


df1 = pd.read_csv('./contest/H-S_seq_1.csv')
df2 = pd.read_csv('./contest/H-S_seq_2.csv')
df3 = pd.read_csv('./contest/H-S_seq_3.csv')
df4 = pd.read_csv('./contest/H-S_seq_4.csv')
df5 = pd.read_csv('./contest/H-S_seq_5.csv')
df6 = pd.read_csv('./contest/H-S_seq_6.csv')
df7 = pd.read_csv('./contest/H-S_seq_7.csv')
df8 = pd.read_csv('./contest/H-S_seq_8.csv')

print('1 :', gap(df1))
print('2 :', gap(df2))
print('3 :', gap(df3))
print('4 :', gap(df4))
print('5 :', gap(df5))
print('6 :', gap(df6))
print('7 :', gap(df7))
print('8 :', gap(df8))

print(df8.iloc[30]['Date'])

# 22~32 경사
