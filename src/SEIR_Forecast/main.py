'''
Author: Junbong Jang
Date 4/29/2020

Load timeseries data, train the model, and forecast

'''


import matplotlib
matplotlib.use('agg')
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 14

import pandas as pd
import numpy as np
from datetime import date
import datetime
import os
import statistics
import math
import pymc3 as pm
from colormap import rgb2hex

from src.EDA.parseJohnsHopkins import johnsHopkinsPopulation, getTzuHsiClusters
from src.SEIR_Forecast import Bayesian_Inference_SEIR
from src.SEIR_Forecast.Bayesian_Inference_plotting import plot_cases
from src.SEIR_Forecast.Bayesian_Inference_SEIR_changepoints import get_change_points
from src.SEIR_Forecast.timeseries_eval import *
from src.SEIR_Forecast.reassignment_helper import *
from src.SEIR_Forecast.data_processing import *
from src.SEIR_Forecast.visualizer import *
from src.SEIR_Forecast.SIR import sir_forecast_a_county


def forecast_main(clusters, cases_df, vel_cases_df, population_df, cluster_mode, init_cluster_num, max_cluster_num, initial_date,
                  final_date, final_change_date, num_days_future, dataset_final_date, run_mode, root_save_path):
    rmse_per_cluster_list = []
    total_re_per_cluster_list = []
    mean_rsquared_per_cluster_list = []
    mape_per_cluster_list = []
    wape_per_cluster_list = []

    mse_per_county_per_cluster_list = []
    re_per_county_per_cluster_list = []
    rsquared_per_county_per_cluster_list = []
    mape_per_county_per_cluster_list = []
    wape_per_county_per_cluster_list = []

    unclustered_rmse_per_cluster_list = []
    unclustered_mse_per_county_per_cluster_list = []
    unclustered_total_re_per_cluster_list = []
    unclustered_re_per_county_per_cluster_list = []

    initial_date = datetime.datetime.strptime(f'{initial_date}/2020', '%m/%d/%Y')
    final_date = datetime.datetime.strptime(f'{final_date}/2020', '%m/%d/%Y')
    dataset_final_date = datetime.datetime.strptime(f'{dataset_final_date}/2020', '%m/%d/%Y')

    cluster_colors = {1: rgb2hex(255,0,0), 2: rgb2hex(255,111,0), 3: rgb2hex(255,234,0), 4: rgb2hex(151,255,0), 5: rgb2hex(44,255,150), 6: rgb2hex(0,152,255), 7:rgb2hex(0,25,255)}

    # cluster id 0 was not clustered by Tzu Hsi but I still use it
    for cluster_id in range(init_cluster_num, max_cluster_num):
        if cluster_mode == 'unclustered':
            cluster_id = 'All'
            chosen_cluster_series = clusters
        else:
            chosen_cluster_series = clusters[clusters == cluster_id]
        cluster_counties = chosen_cluster_series.index.tolist()

        print('-----------------------------')
        print('Cluster ID: ', cluster_id)
        # ------------- Create save folders --------------
        cluster_save_path = root_save_path + f'/cluster_{cluster_id}/'
        if os.path.isdir(cluster_save_path) is False:
            os.mkdir(cluster_save_path)
        cluster_all_save_path = root_save_path + f'/cluster_All/'

        # -------------- Data Preprocessing --------------
        cluster_cases_df, proc_population_series = preprocess_dataset(cases_df.copy(), population_df.copy(), cluster_counties)
        cluster_vel_cases_df, _ = preprocess_dataset(vel_cases_df.copy(), population_df.copy(), cluster_counties)

        cluster_cases_df, current_cumulative_cases_df, future_cumulative_cases_df, old_cumulative_infected_cases_series, date_begin_sim, num_days_sim = \
            process_date(cluster_cases_df, initial_date, final_date, dataset_final_date, num_days_future)
        cluster_vel_cases_df, current_vel_cases_df, future_vel_cases_df, _, _, _ = \
            process_date(cluster_vel_cases_df, initial_date, final_date, dataset_final_date, num_days_future)


        current_cumulative_cases_series = current_cumulative_cases_df.sum(axis=1)
        current_vel_cases_series = current_vel_cases_df.sum(axis=1)
        cluster_total_population = proc_population_series.sum()
        future_cumulative_cases = future_cumulative_cases_df.sum(axis=1)[-1]

        print('old_cumulative_infected_cases_series:', old_cumulative_infected_cases_series)
        print('Cumulative future cases:', future_cumulative_cases)
        print('population:', cluster_total_population)
        print('Remaining population:', cluster_total_population - future_cumulative_cases)

        visualize_trend_with_r_not(cluster_id, cluster_vel_cases_df, cluster_save_path)

        # --------------- Get SIR Model -----------------
        # convert cumulative infected to daily total infected cases
        current_total_cases_series = current_cumulative_cases_series - old_cumulative_infected_cases_series.sum()
        future_total_cases_df = future_cumulative_cases_df - old_cumulative_infected_cases_series

        day_1_cumulative_infected_cases = current_cumulative_cases_series[0]
        S_begin_beta = cluster_total_population - day_1_cumulative_infected_cases
        I_begin_beta = current_total_cases_series[0]  # day 1 total infected cases

        print('day_1_cumulative_infected_cases: ', day_1_cumulative_infected_cases)
        print('S_begin_beta: ', S_begin_beta)
        print('I_begin_beta: ', I_begin_beta)

        change_points = get_change_points(final_date, final_change_date, cluster_id)
        sir_model = Bayesian_Inference_SEIR.SIR_with_change_points(S_begin_beta,
                                                                   I_begin_beta,
                                                                   current_vel_cases_series.to_numpy(),  # current_total_cases_series.to_numpy(),
                                                                   change_points_list=change_points,
                                                                   date_begin_simulation=date_begin_sim,
                                                                   num_days_sim=num_days_sim,
                                                                   diff_data_sim=0,
                                                                   N=cluster_total_population)

        # ---------- Estimate Parameters for SIR model ------------
        if run_mode == 'train':
            Bayesian_Inference_SEIR.run(sir_model, N_SAMPLES=10000, cluster_save_path=cluster_save_path)

        elif run_mode == 'eval':
            trace = pm.load_trace(cluster_save_path + 'sir_model.trace', model=sir_model)
            susceptible_series = proc_population_series - current_cumulative_cases_df.loc[final_date]

            # ---------- Forecast using unclustered data ------------------
            t = range(len(future_vel_cases_df.values))
            if cluster_mode == 'clustered':
                lambda_t, μ = np.load(cluster_all_save_path + 'SIR_params.npy', allow_pickle=True)
                beta, gamma = lambda_t[-1], μ[0]
                print('beta, gamma', beta, gamma)
                cluster_all_vel_case_forecast = sir_forecast_a_county(susceptible_series.sum(), moving_average_from_df(current_vel_cases_df).sum(),
                                                             cluster_total_population, t, beta, gamma, '', '')
            else:
                cluster_all_vel_case_forecast = None

            # ----------- Forecast using clustered data ------------------
            lambda_t, μ = np.load(cluster_save_path + 'SIR_params.npy', allow_pickle=True)
            beta, gamma = lambda_t[-1], μ[0]
            print('beta, gamma', beta, gamma)
            cluster_forecast_I0 = np.mean(trace['new_cases'][:, len(current_vel_cases_series)], axis=0)

            cluster_vel_case_forecast = sir_forecast_a_county(susceptible_series.sum(), cluster_forecast_I0,
                                                         cluster_total_population, t, beta, gamma, '', '')

            # ----------- Forecast Visualization ---------------
            # reorder cluster id to 1,2,3,4,5,6,7 based on severity (rising trend)
            if cluster_id == 0:
                cluster_id = 7
            elif cluster_id == 2:
                cluster_id = 1
            elif cluster_id == 3:
                cluster_id = 3
            elif cluster_id == 4:
                cluster_id = 2
            elif cluster_id == 6:
                cluster_id = 5
            elif cluster_id == 7:
                cluster_id = 6
            elif cluster_id == 8:
                cluster_id = 4

            if cluster_id in cluster_colors.keys():
                cluster_color = cluster_colors[cluster_id]
            else:
                cluster_color = 'black'
            plot_cases(cluster_id, cluster_color, trace, current_vel_cases_series, future_vel_cases_df, cluster_vel_case_forecast,
                    cluster_all_vel_case_forecast, date_begin_sim, diff_data_sim=0, num_days_future=num_days_future, cluster_save_path=cluster_save_path)

            # ---------- Evaluation per county -----------
            cluster_mse_dict, cluster_re_dict, cluster_rsquared_dict, cluster_mape_dict, cluster_wape_dict = \
                eval_per_cluster(susceptible_series, moving_average_from_df(current_vel_cases_df), future_vel_cases_df, proc_population_series, num_days_future,cluster_save_path)

            if cluster_mode == 'unclustered':
                for cluster_id in range(0, max_cluster_num):
                    local_mse_list = []
                    local_re_list = []
                    chosen_cluster_series = clusters[clusters == cluster_id]
                    cluster_counties = chosen_cluster_series.index.tolist()
                    for a_county in cluster_counties:
                        if a_county in cluster_mse_dict:
                            local_mse_list.append(cluster_mse_dict[a_county])
                        if a_county in cluster_re_dict:
                            local_re_list.append(cluster_re_dict[a_county])
                    unclustered_rmse_per_cluster_list.append(math.sqrt(statistics.mean(local_mse_list)))
                    unclustered_mse_per_county_per_cluster_list.append(local_mse_list)
                    unclustered_total_re_per_cluster_list.append(statistics.mean(local_re_list))
                    unclustered_re_per_county_per_cluster_list.append(local_re_list)

            elif cluster_mode == 'clustered':
                rmse_per_cluster_list.append(math.sqrt(statistics.mean(cluster_mse_dict.values())))
                total_re_per_cluster_list.append(statistics.mean(cluster_re_dict.values()))
                mean_rsquared_per_cluster_list.append(statistics.mean(cluster_rsquared_dict.values()))
                # mape_per_cluster_list.append(statistics.mean(cluster_mape_dict.values()))
                wape_per_cluster_list.append(statistics.mean(cluster_wape_dict.values()))

                mse_per_county_per_cluster_list.append(list(cluster_mse_dict.values()))
                re_per_county_per_cluster_list.append(list(cluster_re_dict.values()))
                rsquared_per_county_per_cluster_list.append(list(cluster_rsquared_dict.values()))
                mape_per_county_per_cluster_list.append(list(cluster_mape_dict.values()))
                wape_per_county_per_cluster_list.append(list(cluster_wape_dict.values()))

        if cluster_mode == 'unclustered':
            break  # only run once for unclustered dataset

    return rmse_per_cluster_list, mse_per_county_per_cluster_list, mean_rsquared_per_cluster_list, rsquared_per_county_per_cluster_list, \
           mape_per_cluster_list, mape_per_county_per_cluster_list, wape_per_cluster_list, wape_per_county_per_cluster_list, \
           total_re_per_cluster_list, re_per_county_per_cluster_list, \
           unclustered_rmse_per_cluster_list, unclustered_mse_per_county_per_cluster_list, \
           unclustered_total_re_per_cluster_list, unclustered_re_per_county_per_cluster_list


if __name__ == "__main__":
    # Fitting the SEIR model to the data and estimating the parameters with the cluster id.
    dataset_final_date = '8/1'
    cluster_type = "no_constants"
    run_mode = 'eval'
    # cluster_mode = 'clustered'

    # initial_date_list = ['3/15', '4/1', '4/15', '5/1', '5/15']
    # final_date_list = ['4/30', '5/15', '5/31', '6/15', '6/30']
    # final_change_date_list = [datetime.datetime(2020, 4, 15), datetime.datetime(2020, 4, 30), datetime.datetime(2020, 5, 15), datetime.datetime(2020, 5, 31), datetime.datetime(2020, 6, 15)]


    initial_date_list = ['5/1']
    final_date_list = ['6/15']
    final_change_date_list = [datetime.datetime(2020, 6, 5)]

    # initial_date_list = ['4/1']
    # final_date_list = ['5/15']
    # final_change_date_list = [datetime.datetime(2020, 5, 5)]

    for initial_date, final_date, final_change_date in zip(initial_date_list, final_date_list, final_change_date_list):
        # load data
        initial_clusters = getTzuHsiClusters(column_date=f"{initial_date}~{final_date}", cluster_type=cluster_type)
        max_cluster_num = len(initial_clusters.unique())
        cases_df = pd.read_csv(f'../../generated/us_cases_counties.csv', header=0, index_col=0)
        vel_cases_df = pd.read_csv(f'../../generated/us_velocity_cases_counties.csv', header=0, index_col=0)
        population_df = johnsHopkinsPopulation()

        # set save path
        date_info = f'{initial_date.replace("/","-")}_{final_date.replace("/","-")}_{dataset_final_date.replace("/","-")}_{final_change_date.strftime("%m-%d")}_{cluster_type}'
        root_save_path = f'../../generated/plots/{date_info}/'
        if os.path.isdir(root_save_path) is False:
            os.mkdir(root_save_path)

        # initial Parameters
        reassigned_clusters = initial_clusters
        REASSIGN_COUNTER_MAX = 1
        reassign_counter_init = 0  # to load cluster data from intermediate reassign num
        init_cluster_num = 0

        if run_mode == 'eval':
            max_cluster_num = max_cluster_num - 1  # remove cluster 11 as a outlier

        # if cluster_mode == 'unclustered':
        #     REASSIGN_COUNTER_MAX = 1

        # root_save_path = f'../../generated/plots/{date_info}/reassign_{reassign_counter_init-1}/'
        # reassigned_clusters = pd.read_csv(root_save_path + f'clusters.csv', header=0, index_col=0)
        # reassigned_clusters = reassigned_clusters.iloc[:, 0]
        # reassigned_clusters = reassign_county(reassigned_clusters, max_cluster_num, cases_df, population_df, initial_date, final_date, num_days_future, root_save_path)

        # -------------- Run Reassignment Model ------------------------
        for reassign_counter in range(reassign_counter_init, REASSIGN_COUNTER_MAX):
            print('reassign_counter: ', reassign_counter)
            if reassign_counter < REASSIGN_COUNTER_MAX - 1:
                num_days_future = 7  # only validation set
            else:
                num_days_future = 14  # validation + test set
                
            root_save_path = f'../../generated/plots/{date_info}/reassign_{reassign_counter}/'
            if run_mode == 'eval':  # load reassigned data
                reassigned_clusters = pd.read_csv(root_save_path + f'clusters.csv', header=0, index_col=0)
                reassigned_clusters = reassigned_clusters.iloc[:, 0]
            elif run_mode == 'train':
                if os.path.isdir(root_save_path) is False:
                    os.mkdir(root_save_path)
                reassigned_clusters.to_csv(root_save_path + f'clusters.csv')  # save current county assignment to clusters
            print(reassigned_clusters)

            # ------------ Forecast -----------------
            rmse_per_cluster_list, mse_per_county_per_cluster_list, mean_rsquared_per_cluster_list, rsquared_per_county_per_cluster_list, \
             mape_per_cluster_list, mape_per_county_per_cluster_list, wape_per_cluster_list, wape_per_county_per_cluster_list, \
            total_re_per_cluster_list, re_per_county_per_cluster_list, \
            _, _, _, _ = forecast_main(reassigned_clusters, cases_df, vel_cases_df, population_df, 'clustered',
                                       init_cluster_num, max_cluster_num, initial_date, final_date, final_change_date,
                                       num_days_future, dataset_final_date, run_mode, root_save_path)

            _, _, _, _, \
            _, _, _, _, \
            _, _, \
            unclustered_rmse_per_cluster_list, unclustered_mse_per_county_per_cluster_list, \
            unclustered_total_re_per_cluster_list, unclustered_re_per_county_per_cluster_list \
                = forecast_main(reassigned_clusters, cases_df, vel_cases_df, population_df, 'unclustered',
                                init_cluster_num, max_cluster_num, initial_date, final_date, final_change_date,
                                num_days_future, dataset_final_date, run_mode, root_save_path)

            # ------------------------------
            if run_mode == 'train':
                if reassign_counter < REASSIGN_COUNTER_MAX - 1:
                    reassigned_clusters = reassign_county(reassigned_clusters, max_cluster_num, cases_df, vel_cases_df,
                                                          population_df, initial_date, final_date,
                                                          num_days_future, root_save_path)

            # ------------ Evaluation ---------------
            if run_mode == 'eval':

                # ------------------ for clustered dataset -----------------------
                clustered_average_mse = average_from_list_of_list(mse_per_county_per_cluster_list)
                clustered_average_rmse = round(math.sqrt(clustered_average_mse), 3)
                average_of_rsquared = round(average_from_list_of_list(rsquared_per_county_per_cluster_list), 3)
                average_of_mape = round(average_from_list_of_list(mape_per_county_per_cluster_list), 3)
                average_of_wape = round(average_from_list_of_list(wape_per_county_per_cluster_list), 3)

                histogram_clusters(reassigned_clusters, max_cluster_num, root_save_path)

                violin_eval_clusters(mse_per_county_per_cluster_list, 'MSE', root_save_path)
                violin_eval_clusters(rsquared_per_county_per_cluster_list, 'R^2', root_save_path)
                violin_eval_clusters(mape_per_county_per_cluster_list, 'MAPE', root_save_path)
                violin_eval_clusters(wape_per_county_per_cluster_list, 'WAPE', root_save_path)

                bar_eval_clusters(max_cluster_num, rmse_per_cluster_list, clustered_average_rmse, 'RMSE', 'clustered',
                                  cluster_type, root_save_path)
                bar_eval_clusters(max_cluster_num, mean_rsquared_per_cluster_list, average_of_rsquared, 'R^2', 'clustered',
                                  cluster_type, root_save_path)
                # bar_eval_clusters(max_cluster_num, mape_per_cluster_list, average_of_mape, 'MAPE', 'clustered',
                #                   cluster_type, root_save_path)
                bar_eval_clusters(max_cluster_num, wape_per_cluster_list, average_of_wape, 'WAPE', 'clustered',
                                  cluster_type, root_save_path)

                # --------------- unclustered ---------------------
                unclustered_average_mse = average_from_list_of_list(unclustered_mse_per_county_per_cluster_list)
                unclustered_average_rmse = round(math.sqrt(unclustered_average_mse), 3)
                bar_eval_clusters(max_cluster_num, unclustered_rmse_per_cluster_list, unclustered_average_rmse, 'RMSE', 'unclustered',
                                  cluster_type, root_save_path)

                # ---- clustered and unclustered -----
                print('----------------')

                clustered_average_re = average_from_list_of_list(re_per_county_per_cluster_list)
                unclustered_average_re = average_from_list_of_list(unclustered_re_per_county_per_cluster_list)
                clustered_average_re = round(clustered_average_re, 3)
                unclustered_average_re = round(unclustered_average_re, 3)

                bar_eval_clusters_compare(rmse_per_cluster_list, unclustered_rmse_per_cluster_list, clustered_average_rmse, unclustered_average_rmse, 'RMSE', root_save_path)
                bar_eval_clusters_compare(total_re_per_cluster_list, unclustered_total_re_per_cluster_list, clustered_average_re, unclustered_average_re, 'Relative Error', root_save_path)