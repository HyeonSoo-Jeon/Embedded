

![neural Prophet](https://user-images.githubusercontent.com/78081827/213059043-6a9ac67b-a252-46e5-b43f-ceb2c866635b.png)
<br><br>

# Neural Proepht

<br>
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

## Modelling Trend
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





