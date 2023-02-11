from neuralprophet import NeuralProphet
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 분류 값 반환 함수


def sortData(df):
    cnt = 0
    valueSum = 0
    prev_value = df.iloc[21]['S']
    for _, row in df[22:30].iterrows():
        now_value = row['S']
        valueSum += (now_value-prev_value)**2
        cnt += 1
        prev_value = now_value
    return valueSum/cnt

# Nueral Prophet을 위한 데이터 처리
# [ds, y, H]


def makeNPData(df, ID):
    # column이름 변경
    df.rename(columns={'Date': 'ds', 'S': 'y'}, inplace=True)
    # 날짜 보간
    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index('ds', inplace=True)
    df_daily = df.resample('D').mean()
    # H 보간
    df_daily['H'].fillna(method='ffill', inplace=True)
    df = df_daily.reset_index()
    # data별 ID부여
    df['ID'] = ID
    df = df[['ds', 'y', 'H', 'ID']]
    return df

# RMSE함수


def RMSE(data):
    cnt = 0
    SE = 0
    for _, row in data.dropna().iterrows():
        cnt += 1
        actual_value = row['y']
        predict_value = row['yhat1']
        SE += (actual_value-predict_value)**2
    return_RMSE = (SE/cnt)**(1/2)
    return return_RMSE


# Data 불러오기
df1 = pd.read_csv('H-S_seq_1.csv')
df2 = pd.read_csv('H-S_seq_2.csv')
df3 = pd.read_csv('H-S_seq_3.csv')
df4 = pd.read_csv('H-S_seq_4.csv')
df5 = pd.read_csv('H-S_seq_5.csv')
df6 = pd.read_csv('H-S_seq_6.csv')
df7 = pd.read_csv('H-S_seq_7.csv')
df8 = pd.read_csv('H-S_seq_8.csv')

NPData1 = makeNPData(df1, ID='data1')
NPData2 = makeNPData(df2, ID='data2')
NPData3 = makeNPData(df3, ID='data3')
NPData4 = makeNPData(df4, ID='data4')
NPData5 = makeNPData(df5, ID='data5')
NPData6 = makeNPData(df6, ID='data6')
NPData7 = makeNPData(df7, ID='data7')
NPData8 = makeNPData(df8, ID='data8')

DataSet1 = pd.concat((NPData2, NPData5, NPData6, NPData7, NPData8))
DataSet2 = pd.concat((NPData3, NPData4))

# Neural Prophet 모델 생성
# epoch와 batch_size는 neural prophet에서 최적의 값을 찾아줌
Model1 = NeuralProphet(learning_rate=0.1,
                       yearly_seasonality=False,
                       weekly_seasonality=False,
                       unknown_data_normalization=True,
                       newer_samples_weight=4)
# 회귀 변수 추가
Model1 = Model1.add_future_regressor(name='H', normalize=True)
Model1.fit(DataSet1)

Model2 = NeuralProphet(learning_rate=0.1,
                       yearly_seasonality=False,
                       weekly_seasonality=False,
                       unknown_data_normalization=True,
                       newer_samples_weight=4)
# 회귀 변수 추가
Model2 = Model2.add_future_regressor(name='H', normalize=True)
Model2.fit(DataSet2)


#-------------------------------- TEST-------------------------------- #

# 예측 data 추가
filename = 'filename'           # (수정) 검사 data 이름
test_df = pd.read_csv(filename)
test_NPData = makeNPData(test_df.copy(), ID='test')

# 예측 data 모델 판별
print(test_df)
sortValue = sortData(test_df)

if sortValue < 3:
    forecast = Model1.predict(test_NPData)
else:
    forecast = Model2.predict(test_NPData)

# 예측 확인
# print(forecast)도 가능
result = forecast[['ds', 'y', 'yhat1']]
print(result)

plt.scatter(result['ds'], result['y'],
            label='Actual Data', s=10, color='black')
plt.plot(result['ds'], result['yhat1'], label='Predicted Data')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('S')
plt.legend()
plt.show()

# RMSE
print(f'RMSE of Test Data : {RMSE(forecast)}')
