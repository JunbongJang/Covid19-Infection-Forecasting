'''
Author Junbong Jang
9/4/2020

Metrics to evaluate timeseries forecast

'''

from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

def rmse_eval(future_cases_obs, median_cases_forecast):
    rmse = sqrt(mean_squared_error(future_cases_obs, median_cases_forecast))
    return rmse
    

def rmse_custom_eval(future_cases_obs, median_cases_forecast):
    mse = mse_eval(future_cases_obs, median_cases_forecast)
    return sqrt(mse)
    

def mse_eval(future_cases_obs, median_cases_forecast):
    errors = future_cases_obs - median_cases_forecast
    errors_num = len(errors)
    
    squared_error_sum = 0
    for error in errors:
        squared_error = error ** 2
        squared_error_sum = squared_error_sum + squared_error
    
    return squared_error_sum/errors_num


if __name__ == "__main__":
    save_path = '../../generated/plots/3-15_4-30_8-12_04-15_constants/cluster_0/cases_forecast.npy'
    cases_forecast = np.load(save_path)
    median_cases_forecast = np.median(cases_forecast, axis=-1)
    print(median_cases_forecast)