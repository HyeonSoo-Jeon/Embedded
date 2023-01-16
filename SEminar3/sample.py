import Prophet
import pd

easter_holiday = 'ha'
thanksgiving_holiday = 'ha'
christmas_holiday = 'ha'


lockdowns = pd.DataFrame([
    {'holiday': 'lockdown_1', 'ds': '2020-03-21',
        'lower_window': 0, 'ds_upper': '2020-06-06'},
    {'holiday': 'lockdown_2', 'ds': '2020-07-09',
        'lower_window': 0, 'ds_upper': '2020-10-27'},
    {'holiday': 'lockdown_3', 'ds': '2021-02-13',
        'lower_window': 0, 'ds_upper': '2021-02-17'},
    {'holiday': 'lockdown_4', 'ds': '2021-05-28',
        'lower_window': 0, 'ds_upper': '2021-06-10'},
])
for t_col in ['ds', 'ds_upper']:
    lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
lockdowns['upper_window'] = (lockdowns['ds_upper'] - lockdowns['ds']).dt.days
holidays = pd.concat(
    [easter_holiday, thanksgiving_holiday, christmas_holiday, lockdowns])

prophet = Prophet(holidays=holidays,
                  seasonality_mode='multiplicative',
                  seasonality_prior_scale=20,
                  changepoint_prior_scale=0.1,
                  holidays_prior_scale=10)
