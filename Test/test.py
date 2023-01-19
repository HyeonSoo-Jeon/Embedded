# Random Dataset Preparation
import pandas
import numpy
from prophet import Prophet
import random
random.seed(a=1)

df = pandas.DataFrame(data=None, columns=['ds', 'y', 'ex'], index=range(50))
datelist = pandas.date_range(pandas.datetime.today(), periods=50).tolist()

y = numpy.random.normal(0, 1, 50)
ex = numpy.random.normal(0, 2, 50)

df['ds'] = datelist
df['y'] = y
df['ex'] = ex

# Model
prophet_model = Prophet(seasonality_prior_scale=0.1)
Prophet.add_regressor(prophet_model, 'ex')
prophet_model.fit(df)
prophet_forecast_step = prophet_model.make_future_dataframe(periods=1)

# Result-df
prophet_x_df = pandas.DataFrame(
    data=None, columns=['Date_x', 'Res'], index=range(int(len(y))))

# Error
prophet_x_df.iloc[0, 1] = prophet_model.predict(
    prophet_forecast_step).iloc[0, 0]
