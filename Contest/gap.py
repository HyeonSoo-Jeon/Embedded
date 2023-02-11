import pandas as pd
import matplotlib.pyplot as plt


def gap(df):
    l = 0
    s = 0
    before = 0
    for idx, row in df[19:].iterrows():
        l += 1
        s += (row['S']-before)**4
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

print(gap(df1))
print(gap(df2))
print(gap(df3))
print(gap(df4))
print(gap(df5))
print(gap(df6))
print(gap(df7))
print(gap(df8))
