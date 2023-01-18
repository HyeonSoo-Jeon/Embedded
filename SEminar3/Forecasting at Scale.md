# Prophet

fbprophet에서 package 명이 prophet으로 바뀜
python 3.11에서 작동하지 않아, 3.7로 버전다운시키고 사용

pip install prophet후 plotly를 재설치 해야함


ds와 y의 column을 가져야 한다.
ds의 경우 yyyy-mm-dd (hh:mm:ss)형식을 맞추는 것이 좋음


``` python
m = Prophet()
m.add_country_holidays(country_name='KR')
m.fit(df)
```
Prophet Parameter
- seasonality_mode='multiplicative' : 곱셈 계절성
- holidays_prior_scale=0.05 : 휴일 계절성(default = 10)
- growth='flat' : 일정한 추세 성장률

<br>

## 너무 극단적인 값

<br>
값이 너무 극단적인 경우 NA값으로 만들어 학습에서 반영시키지 않는다.

```python
df.loc[(df['ds'] > '2015-06-01') & (df['ds'] < '2015-06-30'), 'y'] = None
m = Prophet().fit(df)
fig = m.plot(m.predict(future))
```
<br>

## Logistic 성장 모형
<br>

```python
df['cap']=8.5
m=Prophet(growth='logistic')

# 미래 예측
future = m.make_future_dataframe(periods=1826)
future['cap'] = 8.5
fcst = m.predict(future)
fig = m.plot(fcst)
```

역으로 뒤집기

``` python
df['y'] = 10 - df['y']
df['cap'] = 6
df['floor'] = 1.5
future['cap'] = 6
future['floor'] = 1.5
m = Prophet(growth='logistic')
m.fit(df)
fcst = m.predict(future)
fig = m.plot(fcst)
```
<br>

## Holidays 추가

<br>
휴일의 정보를 가진 데이터프레임 

* lower_window : 전날에도 영향을 미치는가, 음수 ex) -1 : 하루 전까지 영향
* upper_window : 다음날에도 영향을 미치는가, 양수
* ds_upper : ds~ds_upper까지, 'yyyy-mm-dd'

<br>

``` python
holidays = pd.DataFrame({
    'holiday' : 'name',
    'ds' : pd.to_datetime(['2020-01-13', '2020-01-24',
                           '2021-01-13', '2021-01-24']),
    'lower_window' : 0,
    'upper_window' : 1,
})

m = Prophet(holidays = holidays)
m.add_country_holidays(country_name='KR')
```

여러 holidays

``` python
lockdowns = pd.DataFrame([
    {'holiday': 'lockdown_1', 'ds': '2020-03-21', 'lower_window': 0, 'ds_upper': '2020-06-06'},
    {'holiday': 'lockdown_2', 'ds': '2020-07-09', 'lower_window': 0, 'ds_upper': '2020-10-27'},
    {'holiday': 'lockdown_3', 'ds': '2021-02-13', 'lower_window': 0, 'ds_upper': '2021-02-17'},
    {'holiday': 'lockdown_4', 'ds': '2021-05-28', 'lower_window': 0, 'ds_upper': '2021-06-10'},
])
for t_col in ['ds', 'ds_upper']:
    lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days
```

pandas.concat함수로 동일한 DataFrame을 합칠 수 있음

```python
playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                        '2010-01-24', '2010-02-07', '2011-01-08',
                        '2013-01-12', '2014-01-12', '2014-01-19',
                        '2014-02-02', '2015-01-11', '2016-01-17',
                        '2016-01-24', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))
```
### 국가 목록
---

Brazil (BR), Indonesia (ID), India (IN), Malaysia (MY), Vietnam (VN), Thailand (TH), Philippines (PH), Pakistan (PK), Bangladesh (BD), Egypt (EG), China (CN), and Russian (RU), Korea (KR), Belarus (BY), and United Arab Emirates (AE)

---

적용된 휴일 목록 확인

``` python
m.train_holiday_names
```
휴일 효과 확인
```python
forecast[(forecast['playoff'] + forecast['superbowl']).abs() > 0][
        ['ds', 'playoff', 'superbowl']][-10:]
```
<br>

## 계절성 푸리에 차수

기본 푸리에 차수는 10이다. 주로 적절하지만 더 높은 빈도의 변화에 바꿀 필요가 있다. 하지만 과적합으로 이어질 수 있다. 2N변수이다.

```python
from prophet.plot import plot_yearly
m = Prophet(yearly_seasonality=20).fit(df)
a = plot_yearly(m)
```

주간 계절성에 대해 푸리에 차수3을 사용하고 연간 계절성에 대해 10을 사용한다.
주간 계절성을 월간 계절성으로 바꾸기

```python
m = Prophet(weekly_seasonality=False)
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
forecast = m.fit(df).predict(future)
fig = m.plot_components(forecast)
```
<br>

## 성수기 비성수기 구분하기
<br>

```python
def is_nfl_season(ds):
    date = pd.to_datetime(ds)
    return (date.month > 8 or date.month < 2)

df['on_season'] = df['ds'].apply(is_nfl_season)
df['off_season'] = ~df['ds'].apply(is_nfl_season)

m = Prophet(weekly_seasonality=False)
m.add_seasonality(name='weekly_on_season', period=7, fourier_order=3, condition_name='on_season')
m.add_seasonality(name='weekly_off_season', period=7, fourier_order=3, condition_name='off_season')

future['on_season'] = future['ds'].apply(is_nfl_season)
future['off_season'] = ~future['ds'].apply(is_nfl_season)
forecast = m.fit(df).predict(future)
fig = m.plot_components(forecast)
```
<br>

## 예측하기
<br>
예측을 어디까지 진행할 것인지 기간을 정한다.

<br>

``` python
future = m.make_future_dataframe(periods=365)
future.tail()
```

make_future_dataframe을 이용해서 쉽게 만들 수 있다. periods에 원하는 기간을 넣어서 생성하면 된다.

predict를 이용해서 예측하면 되는데, forecast에 많은 정보가 들어간다.

``` python
forecase = m.predict(future)
forecast[['ds', 'yhat','yhat_lower', 'yhat_upper']].tail()

```

예측 데이터
```python
m.plot(forecast)
```
구성 요소
```python
m.plot_components(forecast)
```

대화형 그림

```python
from prophet.plot import plot_plotly, plot_components_plotly

plot_plotly(m, forecast)
plot_components_plotly(m, forecast)
```
<br>

## ChangePoint 감지
<br>

```python

from prophet.plot import add_changepoints_to_plot
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)

```

### 추세의 유연성 조절

추세 변화가 과적합(유연성이 너무 높음) 또는 과소적합(유연성이 충분하지 않음)인 경우 입력 인수를 사용하기 전에 희소 강도를 조정할 수 있습니다. (changepoint_prior_scale) 기본적으로 이 매개변수는 0.05로 설정됩니다.

```python

# 추세 유연성 향상
m = Prophet(changepoint_prior_scale=0.5)
forecast = m.fit(df).predict(future)
fig = m.plot(forecast)

# 추세 유연성 감소
m = Prophet(changepoint_prior_scale=0.001)
forecast = m.fit(df).predict(future)
fig = m.plot(forecast)

```
<br>

## Changepoint 조정
<br>

```python
m = Prophet(changepoints=['2014-01-01'])
forecast = m.fit(df).predict(future)
fig = m.plot(forecast)
```

<br>

## 모델 저장
<br>

```python
from prophet.serialize import model_to_json, model_from_json

with open('serialized_model.json', 'w') as fout:
    fout.write(model_to_json(m))  # Save model

with open('serialized_model.json', 'r') as fin:
    m = model_from_json(fin.read())  # Load model
```
<Br>

## 적합 모델 업데이트
<Br>


```python
def warm_start_params(m):
    """
    Retrieve parameters from a trained model in the format used to initialize a new Stan model.
    Note that the new Stan model must have these same settings:
        n_changepoints, seasonality features, mcmc sampling
    for the retrieved parameters to be valid for the new model.

    Parameters
    ----------
    m: A trained model of the Prophet class.

    Returns
    -------
    A Dictionary containing retrieved parameters of m.
    """
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        if m.mcmc_samples == 0:
            res[pname] = m.params[pname][0][0]
        else:
            res[pname] = np.mean(m.params[pname])
    for pname in ['delta', 'beta']:
        if m.mcmc_samples == 0:
            res[pname] = m.params[pname][0]
        else:
            res[pname] = np.mean(m.params[pname], axis=0)
    return res

df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
df1 = df.loc[df['ds'] < '2016-01-19', :]  # All data except the last day
m1 = Prophet().fit(df1) # A model fit to all data except the last day


%timeit m2 = Prophet().fit(df)  # Adding the last day, fitting from scratch
%timeit m2 = Prophet().fit(df, init=warm_start_params(m1))  # Adding the last day, warm-starting from m1
```