

![neural Prophet](https://user-images.githubusercontent.com/78081827/213059043-6a9ac67b-a252-46e5-b43f-ceb2c866635b.png)
<br><br>

Prophet + 신경망(Neural Network)

Prophet은 시계열 예측 성능보다는 손쉬운 사용, 예측 결과 해석에 초점을 맞춤

비즈니스적인 시사점이 필요할 때 Prophet은 유용하게 사용된다.

기존의 Prophet보다 52~92% 성능 개선

<br><br>

## $\hat{y}_t = T(t) + S(t) + E(t) + F(t) + A(t) + L(t)$

<br>

- $T(t)$ : 선형함수를 통하여 비선형 트렌트를 예측<br>

- $S(t)$ : 다중 계절성 포착<br>

- $E(t)$ : 이벤트를 모델에 적용<br>

- $F(t)$ : 미래에 알 수 있는 변수들을 회귀 예측에 사용<br>

- $A(t)$ : AR에서 전방향 신경망 네트워크를 적용한 AR-Net사용<br>

- $L(t)$ : 모델 해석에 적용되는 공변량 변수들<br>

<br>

## Sample Code
<br>

```python
from neuralprophet import NeuralProphet
import pandas as pd

data_location = "https://raw.githubusercontent.com/ourownstory/neuralprophet-data/main/datasets/"

df = pd.read_csv(data_location + 'wp_log_peyton_manning.csv')

m = NeuralProphet()
metrics = m.fit(df)
forecast = m.predict(df)

forecasts_plot = m.plot(forecast)       # 예측 시각화
fig_comp = m.plot_components(forecast)  # 구성요소 시각화
fig_param = m.plot_parameters()         # 계수 값 시각화
```
<br>

매번 실행할 때 다른 예측이 나올 수 있으므로 무작위성을 제거할 수 있다.

```python
from neuralprophet import set_random_seed 
set_random_seed(0)
```
<br><br>

# 1. Modelling Trend
<br>

변경 지점을 정의한 Neural Prophet에서 추세 모델링의 예시 코드

```python
m = NeuralProphet(
    n_changepoints=100,
    trend_reg=2,
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
)
metrics = m.fit(df)
forecast = m.predict(df)
```

구성요소 그래프(components plot)는 추세와 잔차만 보여준다. 

![plot_comp_trend_1](https://user-images.githubusercontent.com/78081827/213064945-251968dc-001c-45c0-9fa4-44ca8b3657d4.png)

계수 그래프(coefficient plot)은 100개의 변화점에 해당하는 계수를 표시한다.

![plot_param_trend_1](https://user-images.githubusercontent.com/78081827/213066280-9d645931-23e7-4a35-b17e-6a62e0d127a9.png)
<br><br>

# 2. Modelling Seasonality
<Br>

Neural Prophet의 계절성은 푸리에 항을 사용해서 모델링된다. addictive와 multiplicative 모두에서 계수를 지정할 수 있다.
<br><br>

## Addictive Seasonality
<br>
계절성에 대한 dafault 값은 addictive mode이다.
<br><br> 

```python
m = NeuralProphet()
metrics = m.fit(df)
```

![plot_comp_seasonality_1](https://user-images.githubusercontent.com/78081827/213066499-7b5871eb-de5d-44d0-a6b0-c7146f88046c.png)

위의 그림에서 주간 및 연간 계절 모양을 모두 볼 수 있다. 필요한 계절성이 모델 개발에 명시적으로 명시되어 있지 않기 때문에 Neural Prophet은 데이터에 가능한 모든 계절성을 맞춘다. 모델은 또한 모든 계절성에 대해 원하는 푸리에 항의 수에 기본값을 할당한다. 

아래와 같이 숫자를 지정할 수 있다.

```python
m = NeuralProphet(
    yearly_seasonality=8,
    weekly_seasonality=3
)
```

연간 계절성은 8개의 푸리에 항을 사용하고, 주간 계절 패턴은 3개의 푸리에 항을 사용한다. 푸리에 항을 조절하면서 계절성을 과소적합 또는 과적합 할 수 있다. 다음은 각 계절성에 대해 푸리에 항이 과적합된 예시이다.

```python
m = NeuralProphet(
    yearly_seasonality=16,
    weekly_seasonality=8
)
```

![plot_comp_seasonality_2](https://user-images.githubusercontent.com/78081827/213066502-469f4871-23e4-4a30-8f93-db6b2bdc4116.png)
<br><br>

## Multiplicative Seasonality
<Br><br>
아래와 같이 모드를 명시적으로 설정하여 계절성을 곱셈적으로 모델링할 수도 있다. 이렇게 하면 계절성이 추세와 관련해 배가된다.
<Br><br>

```python
m = NeuralProphet(
    seasonality_mode='multiplicative'
)
```
<br>

## 계절성 정규화
<br>
NeuralProphet의 다른 모든 구성 요소와 마찬가지로 계절성도 정규화할 수 있습니다. 이는 아래와 같이 푸리에 계수를 정규화하여 수행된다. 'seasonality_reg' 매개변수 설정 방법에 대한 자세한 내용은 Hyperparameter Selection 파트에서 다룬다.
<Br><Br>

```python
m = NeuralProphet(
    yearly_seasonality=16,
    weekly_seasonality=8,
    daily_seasonality=False,
    seasonality_reg=1,
)
```
<Br>

# 3. Modelling Auto-Regression
<br>

Neural Prophet에서 'n_lags' parameter에 값을 설정하여 AR-Net을 활성화 할 수 있다.
<Br><br>

```python
m = NeuralProphet(
    n_forecasts=3,
    n_lags=5,
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
)
```

위에서 AR-Net에 5 lags(지연)을 주고, 3개를 예측 받는 시나리오를 만든다. AR-Net을 활성화 하면 future_periods를 예측하는 동안 Neural Prophet객체가 생성될때 명시된 n_forecasts의 값과 같아야 한다. future_periods에 어떤 값을 명시했든, n_forecasts의 값으로 알림과 함께 전환된다. 이것은 AR-Net이 n_forecasts의 출력 사이즈를 가지고 트레이닝 되는 동안 다른 값을 지원하지 않기 때문이다.

components plot은 아래와 같다.
<br><br>

![plot_comp_ar_1](https://user-images.githubusercontent.com/78081827/213070150-fc4bb15f-ae6e-42ea-8436-64cdf95bac5c.png)
<br>

자동 회귀를 별도의 구성 요소로 볼 수 있다.
<br><Br>

![plot_param_ar_1](https://user-images.githubusercontent.com/78081827/213070159-3ee6c73c-735d-4962-bbcb-09781fb9e088.png)

자기 상관(Autoregressive)를 모델링할 때 각 시차의 관련성을 볼 수 있다. AR-Net의 복잡성을 증가시키기 위해 AR-Net에 대한 'num_hidden_layers'를 지정할 수 있다.

```python
m = NeuralProphet(
    n_forecasts=3,
    n_lags=5,
    num_hidden_layers=2,
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False
)
```
<br>

## AR-Net 정규화
<br>

AR-Net에서 정규화는 아래 코드 처럼 'ar_sparsity' 매개변수를 설정해서 이루어진다. 자세한 설명은 Hyper Parameter의 'ar_sparsity'를 참고.

```python
m = NeuralProphet(
    n_forecasts=3,
    n_lags=5,
    num_hidden_layers=2,
    ar_sparsity=0.01,
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False
)
```
<br>

## 특정 예측 단계
<br>

자기 상관을 모델링할 때 다중 입력, 다중 출력 모드로 모델을 만든다. 이 모드에서는 n번째 단계 예측을 강조할 수 있다. 즉, 모델 학습 중 오류를 계산할 때와 예측 플롯을 작정할 때 n번째 단계에서 예측을 구체적으로 볼 수 있다.

```python
m = NeuralProphet(
    n_forecasts=30,
    n_lags=60,
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False
)
m.highlight_nth_step_ahead_of_each_forecast(step_number=m.n_forecasts)
```

'step_number' 매개 변수에 'n_forecasts'보다 작거나 같은 값을 지정할 수 있다. 

예측 플롯에서는 n번째 예측에만 초점을 맞춘다.

![plot_forecast_ar_1](https://user-images.githubusercontent.com/78081827/213071889-33d75de0-d9f8-469f-a1dd-4e75e74f556a.png)

<br><br>

# 4. Modelling Lagged Regressors
<br>

NeuralProphet 개발의 현재 상태에서 Lagged Regressor 지원은 AR-Net이 활성화된 경우에만 사용할 수 있다. Feed-Forward 신경망을 사용하여 내부적으로 유사한 방식으로 처리되고 n_lags값을 지정해야 하기 때문이다. n_lags단순화를 위해 현재 AR-Net과 Lagged Regressor 모두에 대해 동일한 값을 사용한다 . 따라서 Lagged Regressor를 사용하면 아래와 같이 AR-Net과 유사하게 NeuralProphet 객체가 인스턴스화된다.

```python
m = NeuralProphet(
    n_forecasts=3,
    n_lags=5,
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
)
```

모델을 피팅할 때 fit함수에 제공된 데이터 프레임에는 아래와 같이 지연된 회귀 변수에 대한 추가 열이 있어야 한다.


|   | ds         | y       | A       |
|---|------------|---------|---------|
| 0 | 2007-12-10 | 9.59076 | 0.59076 |
| 1 | 2007-12-11 | 8.51959 | 9.05518 |
| 2 | 2007-12-12 | 8.18365 | 8.76468 |
| 3 | 2007-12-13 | 8.07257 | 8.59162 |

이 예제에는 A라는 Lagged Regressor가 있다. 또한 함수 add_lagged_regressor를 호출 하고 필요한 구성(configs)을 제공하여 이러한 Lagged Regressor를 개체에 등록해야 합니다 .

```python
m = m.add_lagged_regressor(names='A')
```

only_last_value함수의 인수 를 설정하여 add_lagged_regressor사용자는 입력 창 내에서 회귀자의 마지막으로 알려진 값만 사용하거나 자동 회귀와 동일한 수의 시차를 사용하도록 지정할 수 있다. 이제 평소와 같이 모델 피팅 및 예측을 수행할 수 있습니다. component plot은 아래와 같아야 한다.

![plot_comp_lag_reg_1](https://user-images.githubusercontent.com/78081827/213072808-830cffc4-3080-43ab-9a68-8ef3924f4662.png)

coefficient plot은 아래와 같다.

![plot_param_lag_reg_1](https://user-images.githubusercontent.com/78081827/213072820-2b8ac6ec-63f1-4b6a-8f21-924eb05a2a2c.png)

<br><br>


# 5. Modelling Events
<br>

종종 문제를 예측할 때 반복되는 특수 이벤트를 고려해야 합니다. 이들은 Neural Prophet에서 지원한다. 이러한 이벤트는 더하기 형식과 곱하기 형식으로 모두 추가할 수 있다.

이벤트 정보를 모델에 제공하기 위해 사용자는 이벤트 날짜에 해당하는 ds열과 지정된 날짜의 이벤트 이름을 포함하는 열이 있는 데이터 프레임을 생성해야 한다. 

```python
playoffs_history = pd.DataFrame({
        'event': 'playoff',
        'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                              '2010-01-24', '2010-02-07', '2011-01-08',
                              '2013-01-12', '2014-01-12', '2014-01-19',
                              '2014-02-02', '2015-01-11', '2016-01-17']),
    })

superbowls_history = pd.DataFrame({
    'event': 'superbowl',
    'ds': pd.to_datetime(['2010-02-07', '2014-02-02']),
    })
history_events_df = pd.concat((playoffs_history, superbowls_history))
```

예측을 위해 모델 교육에 사용되는 이러한 이벤트의 미래 날짜도 제공해야 한다. 모델을 맞추기 위해 이전에 만든 동일한 이벤트 데이터 프레임이나 다음과 같이 새 데이터 프레임에 이를 포함할 수 있다.

```python
playoffs_future = pd.DataFrame({
    'event': 'playoff',
    'ds': pd.to_datetime(['2016-01-21', '2016-02-07'])
})

superbowl_future = pd.DataFrame({
    'event': 'superbowl',
    'ds': pd.to_datetime(['2016-01-23', '2016-02-07'])
})

future_events_df = pd.concat((playoffs_future, superbowl_future))
```

이벤트 데이터 프레임이 생성되면 NeuralProphet객체를 생성하고 이벤트 구성을 추가해야 한다. 이것은 Neural Prophet의 add_events기능을 사용하여 수행된다.

```python
m = NeuralProphet(
        n_forecasts=10,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
m = m.add_events(["superbowl", "playoff"])
```

그런 다음 이전에 생성된 데이터 프레임의 이벤트 데이터를 모델에서 예상하는 이진 입력 데이터로 변환해야 합니다. create_df_with_events함수를 호출함으로써 history_events_df를 변환할 수 있다.

```python
history_df = m.create_df_with_events(df, history_events_df)
```

그 다음, 간단히 생성된 history_df를 모델에서 제공하는 fit함수로 fitting할 수 있다.

```python
metrics = m.fit(history_df)
forcast = m.predict(df=history_df)
```

<제작된 예측은 아래와 같습니다. 10개의 단계별 예측은 yhat1 열에서 사용할 수 있다. 개별 이벤트의 구성 요소는 event_playoff 및 event_superbowl 열에서 사용할 수 있으며 집계된 효과는 events_additive 열에 표시된다.>

예측이 완료되면 아래와 같이 다양한 구성 요소를 그릴 수 있다. 모든 이벤트는 하나의 구성 요소로 플롯된다. 아래는 Additive Events의 예시이다.
<br><Br>

![plot_comp_events_1](https://user-images.githubusercontent.com/78081827/213083361-d755c14c-6857-4213-977d-6d4aa62a3089.png)

coefficient는 아래와 같다.

![plot_param_events_1](https://user-images.githubusercontent.com/78081827/213083364-adee23af-6ae6-4bfb-bc97-5f189b982a10.png)
<br><br>

## Multiplicative Events
<Br>

Multiplicative 모드로 설정하기 위해서는 아래와 같이 명시해줘야한다.

```python
m = m.add_events(["superbowl", "playoff"], mode = "multiplicative")
```

아래의 그림은 Event component를 백분율로 나타낸 것이다.

![plot_comp_events_2](https://user-images.githubusercontent.com/78081827/213083369-8739c517-17a9-48d9-8090-a2c4010f95ca.png)
<br><br>

## Event Windows
<br>

Event Windows를 제공할 수도 있습니다. 이렇게 하면 Neural Prophet의 add_events함수에 적절한 인수 lower_window와 upper_window를 제공하여 특정 이벤트 주변의 날짜를 특별 이벤트로 간주할 수 있다. 기본적으로 이러한 window의 값은 0이며 이는 window가 고려되지 않음을 의미합니다.

```python
m = m.add_events(["superbowl", "playoff"], lower_window=-1, upper_window=1)
```

위 코드에 따르면 superbowl및 playoff이벤트 모두에 대해 이벤트 날짜, 전날 및 다음 날의 삼일의 특별 이벤트가 모델링됩니다. 이들은 아래와 같이 구성 요소 플롯에서 볼 수 있습니다.

![plot_comp_events_3](https://user-images.githubusercontent.com/78081827/213083374-b51a67bd-f407-4191-aabe-c7dee19bde4c.png)

parameter plot에서도 이벤트 이전 및 다음 날의 해당하는 계수가 있다. superbowl과 playoff 둘 다 해당된다.

![plot_param_events_3](https://user-images.githubusercontent.com/78081827/213083376-d469e84e-336d-47a6-9665-0b5cb96a93c3.png)

개별 이벤트에 대해 다른 window를 정의하는 경우 아래와 같이 한다.

```python
m = m.add_events("superbowl", lower_window=-1, upper_window=1)
m = m.add_events("playoff", upper_window=2)
```
<br>

## 국가별 공휴일
<br>

사용자 지정 이벤트 외에도 Neural Prophet 표준 국가별 공휴일도 지원한다. 특정 국가의 공휴일을 추가하려면 add_country_holidays 함수 를 호출하고 국가를 지정하기만 하면 된다. 사용자 지정 이벤트와 유사하게 국가별 공휴일도 additive, multiplicative를 포함할 수 있습니다. 그러나 사용자 지정 이벤트와 달리 window은 모든 국가별 이벤트에 대해 동일합니다.

```python
m = m.add_country_holidays("US", mode="additive", lower_window=-1, upper_window=1)
```

이 예제는 모든 미국 공휴일을 additive형식으로 모델에 추가했다. 개별 이벤트의 계수는 이제 아래와 같다.

![plot_param_events_4](https://user-images.githubusercontent.com/78081827/213083386-7db673f8-c08c-42c4-a539-e4929555c384.png)
<br><br>

## 이벤트 정규화
<br>

이벤트는 계수의 정규화도 지원할 수 있다. 아래와 같이 이벤트 구성을 Neural Prophet객체 에 추가할 때 정규화를 지정할 수 있다.

```python
m = m.add_events(["superbowl", "playoff"], regularization=0.05)
```

개별 이벤트에 대한 정규화도 아래와 같이 서로 다를 수 있다.

```python
m = m.add_events("superbowl", regularization=0.05)
m = m.add_events("playoff", regularization=0.03)
```

국가별 공휴일도 아래와 같이 정규화가 가능하다.

```python
m = m.add_country_holidays("US", mode="additive", regularization=0.05)
```
<br><br>

# 6. Modelling Future Regressors
<br>

*Future Regressor는 전체 예측 기간(예: n_forecasts)에 대해 알려야 한다.*
<br><Br>

미래 회귀 변수는 미래 값을 알고 있는 외부 변수다. 그런 의미에서 Future Regressor는 특별 이벤트와 매우 유사한 기능을 한다.

Training time stamps에 해당하는 이러한 회귀자의 과거 값은 교육 데이터 자체와 함께 제공되어야 한다. 두 개의 dummy regressor를 생성하고 원본 데이터의 롤링 수단을 사용하는 예는 아래를 보자.

```python
df['A'] = df['y'].rolling(7, min_periods=1).mean()
df['B'] = df['y'].rolling(30, min_periods=1).mean()
```

|   | ds         |       y |       A |       B |
|---|------------|--------:|--------:|--------:|
| 0 | 2007-12-10 | 9.59076 | 9.59076 | 9.59076 |
| 1 | 2007-12-11 | 8.51959 | 9.05518 | 9.05518 |
| 2 | 2007-12-12 | 8.18368 | 8.76468 | 8.76468 |
| 3 | 2007-12-13 | 8.07247 | 8.59162 | 8.59162 |
| 4 | 2007-12-14 | 7.89357 | 8.45201 | 8.45201 |

<br>
예측을 수행하기 위해, 회귀자의 미래 값을 제공해야한다.
<br><br>

```python
future_regressors_df = pd.DataFrame(data={'A': df['A'][:50], 'B': df['B'][:50]})
```

|   |       A |       B |
|--:|--------:|--------:|
| 0 | 9.59076 | 9.59076 |
| 1 | 9.05518 | 9.05518 |
| 2 | 8.76468 | 8.76468 |
| 3 | 8.59162 | 8.59162 |
| 4 | 8.45201 | 8.45201 |

회귀자의 미래 값 열만 있는 데이터 프레임이다.

이벤트와 마찬가지로 미래 회귀 변수도 addictive 및 multiplicative 모드로 추가할 수 있다.
<br><Br>

## Additive Future Regressors
<br>

회귀자는 Neural Prophet의 함수 add_future_regressor를 호출하여 객체에 추가된다. 이 작업이 완료되면 학습 데이터의 데이터 프레임과 회귀 값을 함수 fit에 제공하여 모델을 맞출 수 있다.

```python
m = NeuralProphet(
        n_forecasts=10,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )

m = m.add_future_regressor(name='A')
m = m.add_future_regressor(name='B')

metrics = m.fit(df)
forecast = m.predict(df)
```

component plot이다.

```python
fig_comp = m.plot_components(forecast)
```

![plot_comp_future_reg_1](https://user-images.githubusercontent.com/78081827/213088592-81308168-9af7-44bc-86e0-12e17fc7bdab.png)
<br><br>

## Multiplicative Futre Regressors
<Br>

regressor를 추가할 때 모드를 설정해야한다.

```python
m = m.add_future_regressor(name='A', mode="multiplicative")
m = m.add_future_regressor(name='B')
```

위의 예에서 addictive mode와 multiplicative 모드가 모두 있다. A는 multiplicative이고, B는 addictive이다. 피팅 및 예측 프로세스의 다른 단계는 모두 동일하다.

<구성 요소 플롯은 다음과 같습니다. 곱셈 성분이 백분율로 표시되는 덧셈 및 곱셈 회귀 변수에 대한 두 개의 개별 플롯이 있습니다. 같은 방식으로 계수는 아래와 같은 플롯으로 나타납니다.>
<br><br>

## Regularization for Future Regressors
<br>

아래와 같이 미래의 회귀자에 정규화를 추가할 수 있습니다.

```python
m = m.add_future_regressor(name='A', regularization=0.05)
m = m.add_future_regressor(name='B', regularization=0.02)
```

이렇게 하면 개별 회귀자 계수에 희소성이 추가됩니다.
<br><br>

