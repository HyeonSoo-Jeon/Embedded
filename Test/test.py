from itertools import product

from prophet import Prophet

m = Prophet()

param_grid = {
    'model': [m], 'initial': ['730 days', '500 days'], 'period': ['180 days'], 'horizon': ['365 days']
}


def create_grid(param_grid):

    param_grid_list = [param_grid].copy()

    for p in param_grid_list:
        # Always sort the keys of a dictionary, for reproducibility
        items = sorted(p.items())
        if not items:
            yield {}
        else:
            keys, values = zip(*items)
            for v in product(*values):
                params = dict(zip(keys, v))
                yield params


>> > list(create_grid(param_grid))
    [{'model': < fbprophet.forecaster.Prophet at 0x11cd0bba8 > ,
      'initial': '730 days',
      'period': '180 days',
      'horizon': '365 days'},
     {'model': < fbprophet.forecaster.Prophet at 0x11cd0bba8 > ,
      'initial': '500 days',
      'period': '180 days',
      'horizon': '365 days'}]
