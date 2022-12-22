# ARIMA(p,d,q)
# p : 자기 회귀 차수
# d : 차분 차수
# q : 이동 평균 차수

from pandas import read_csv
# from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot


# def parser(x):  # 시간을 표현하는 함수
#     return datetime.strptime('199'+x, '%Y-%m')


series = read_csv('sales.csv', header=0,
                  parse_dates=[0], index_col=0)
# series = read_csv('sales.csv', header=0,
#                   parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit(disp=0)  # 디버그 정보 비활성화
print(model_fit.summary())
residuals = DataFrame(model_fit.resid)  # 모델에 대한 오차 정보 저장
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.decribe())
