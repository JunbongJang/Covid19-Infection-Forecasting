'''
Author Junbong Jang
9/4/2020

Metrics to evaluate timeseries forecast

'''

from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import statistics

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


def eval_per_county(future_vel_cases_df, num_days_future, cluster_save_path):
    cluster_cases_forecast = np.load(cluster_save_path + 'cases_forecast.npy')
    median_cluster_cases_forecast = np.median(cluster_cases_forecast, axis=-1)

    cluster_mse_dict = {}
    future_vel_cases_df = future_vel_cases_df.fillna(0)
    for county in future_vel_cases_df.columns:
        mse = mse_eval(future_vel_cases_df[county].tolist(), median_cluster_cases_forecast)

        cluster_mse_dict[county] = mse
        # all_mse_list.append(mse)
        assert len(future_vel_cases_df[county].tolist()) == num_days_future
        assert len(future_vel_cases_df[county].tolist()) == len(median_cluster_cases_forecast)
    
    return cluster_mse_dict


def eval_a_county(county_future_vel_cases, cluster_save_path):
    cluster_cases_forecast = np.load(cluster_save_path + 'cases_forecast.npy')
    median_cluster_cases_forecast = np.median(cluster_cases_forecast, axis=-1)
    
    county_future_vel_cases = county_future_vel_cases.fillna(0)
    mse = mse_eval(county_future_vel_cases.tolist(), median_cluster_cases_forecast)
    
    return mse


def get_outlier_threshold_from_list(input_list):
    mean = statistics.mean(input_list)
    std = statistics.stdev(input_list)
    upper_outlier_threshold = mean + std*2
    
    return upper_outlier_threshold


if __name__ == "__main__":
    save_path = '../../generated/plots/3-15_4-30_8-12_04-15_constants/cluster_0/cases_forecast.npy'
    cases_forecast = np.load(save_path)
    median_cases_forecast = np.median(cases_forecast, axis=-1)
    print(median_cases_forecast)