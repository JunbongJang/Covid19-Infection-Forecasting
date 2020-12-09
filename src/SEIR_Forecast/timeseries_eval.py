'''
Author Junbong Jang
9/4/2020

Metrics to evaluate timeseries forecast
'''

from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt, isnan, isinf
import numpy as np
import statistics
from scipy import stats
from src.SEIR_Forecast.SIR import sir_forecast_a_county


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


def relative_error_eval(future_cases_obs, median_cases_forecast):
    # errors = (future_cases_obs - median_cases_forecast) / future_cases_obs
    # errors_num = len(errors)
    #
    # squared_error_sum = 0
    # for error in errors:
    #     squared_error = error ** 2
    #     squared_error_sum = squared_error_sum + squared_error

    # error = (sum(future_cases_obs) - sum(median_cases_forecast))**2 / sum(future_cases_obs)**2
    # return error

    y_true, y_pred = np.array(future_cases_obs), np.array(median_cases_forecast)
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)


def r_squared_eval(future_cases_obs, median_cases_forecast):
    correlation_matrix = np.corrcoef(future_cases_obs, median_cases_forecast)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy ** 2
    # if isnan(r_squared):
    #     r_squared = 0

    return r_squared


def mape_eval(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def wape_eval(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sum(np.abs(y_true - y_pred))/ np.sum(y_true) * 100


def get_median_cluster_cases_forecast(cluster_save_path):
    cluster_cases_forecast = np.load(cluster_save_path + 'cases_forecast.npy')
    print('cluster_cases_forecast', cluster_cases_forecast.shape)

    return np.median(cluster_cases_forecast, axis=-1)


def eval_per_cluster(susceptible_series, averaged_infected_series, future_cases_df, proc_population_series, num_days_future, cluster_save_path):

    # replace median forecast below
    median_cluster_cases_forecast = get_median_cluster_cases_forecast(cluster_save_path)
    # with SIR model forecasting per county
    lambda_t, μ = np.load(cluster_save_path + 'SIR_params.npy', allow_pickle=True)
    beta, gamma = lambda_t[-1], μ[0]
    t = range(len(future_cases_df.values))

    cluster_mse_dict = {}
    cluster_mse_relative_dict = {}
    cluster_rsquared_dict = {}
    cluster_mape_dict = {}
    cluster_wape_dict = {}
    future_cases_df = future_cases_df.fillna(0)
    for county in future_cases_df.columns:
        county_case_forecast = sir_forecast_a_county(susceptible_series[county], averaged_infected_series[county], proc_population_series[county], t, beta, gamma, county, cluster_save_path)
        # print(median_cluster_cases_forecast)
        mse, mse_relative, r_squared, mape, wape = eval_a_county(future_cases_df[county].tolist(), county_case_forecast)
        cluster_mse_dict[county] = mse

        if not isnan(mse_relative) and not isinf(mse_relative):
            cluster_mse_relative_dict[county] = mse_relative
        if not isinf(mape):
            cluster_mape_dict[county] = mape
        if not isinf(wape):
            cluster_wape_dict[county] = wape
        if not isnan(r_squared):
            cluster_rsquared_dict[county] = r_squared
            
        assert len(future_cases_df[county].tolist()) == num_days_future
        assert len(future_cases_df[county].tolist()) == len(median_cluster_cases_forecast)

    return cluster_mse_dict, cluster_mse_relative_dict, cluster_rsquared_dict, cluster_mape_dict, cluster_wape_dict


def eval_a_county(county_future_vel_cases, median_cluster_cases_forecast):
    mse = mse_eval(county_future_vel_cases, median_cluster_cases_forecast)
    relative_error = relative_error_eval(county_future_vel_cases, median_cluster_cases_forecast)
    r_squared = r_squared_eval(county_future_vel_cases, median_cluster_cases_forecast)
    mape = mape_eval(county_future_vel_cases, median_cluster_cases_forecast)
    wape = wape_eval(county_future_vel_cases, median_cluster_cases_forecast)

    return mse, relative_error, r_squared, mape, wape


def get_outlier_threshold_from_list(input_list):
    n = len(input_list)
    mean = statistics.mean(input_list)
    std = statistics.stdev(input_list)
    # upper_outlier_threshold = mean + std * 1.645  # 2

    # Student's t statistic, p<0.05%, Single tail
    t_val = stats.t.ppf(1 - 0.05, n)
    upper_outlier_threshold = mean + (std**2)/2 + t_val * (std**2/n + std**4/(2*(n-1))) ** (1/2)

    print('n', n, ' mean', mean, ' std', std, ' t_val', t_val)
    print('upper_outlier_threshold', upper_outlier_threshold)
    print('old threshold', mean + std * 1.645, mean + std * 2)

    return upper_outlier_threshold


def moving_average_from_df(input_df):
    return input_df.iloc[-5:].mean(axis=0)


def average_from_list_of_list(error_county_per_cluster_list):
    all_error_list = [j for sub in error_county_per_cluster_list for j in sub]
    return statistics.mean(all_error_list)


if __name__ == "__main__":
    save_path = '../../generated/plots/3-15_4-30_8-12_04-15_constants/cluster_0/cases_forecast.npy'
    cases_forecast = np.load(save_path)
    median_cases_forecast = np.median(cases_forecast, axis=-1)
    print(median_cases_forecast)