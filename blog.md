보간법

```python
df['ds'] = pd.to_datetime(df['ds'])
df.set_index('ds', inplace = True)
df_daily = df.resample('D').mean()

df_daily['H'].fillna(method='ffill', inplace = True)
# .interpolate()

t1 = df_daily.interpolate(method='polynomial', order = 2)
# t1 = df_daily.interpolate(method='cubic')
df.reset_index()
t1.reset_index()
df.reset_index().plot.scatter(x='ds',y='y', label = 'origin')
t1.reset_index().plot.scatter(x='ds',y='y', label = 'fill')
plt.legend()
plt.show()

```