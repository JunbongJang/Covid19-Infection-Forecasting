'''
Author: Junbong Jang
Date 4/29/2020

Load timeseries data, train the model, and forecast

'''

import pandas as pd
import numpy as np
from datetime import date
import datetime
import os
import statistics 
import math

from src.EDA.parseJohnsHopkins import getTzuHsiClusters, johnsHopkinsPopulation
from src.SEIR_Forecast import Bayesian_Inference_SEIR
from src.SEIR_Forecast.Bayesian_Inference_SEIR_changepoints import get_change_points
from src.SEIR_Forecast.timeseries_eval import *
from src.SEIR_Forecast.cycle_helper import *
from src.SEIR_Forecast.data_processing import *
from src.SEIR_Forecast.visualizer import *


def forecast_main(clusters, cases_df, vel_cases_df, population_df, clusters_num, initial_date, cluster_final_date, final_change_date_list, date_info, num_days_future, forecast_final_date, root_save_path):
    cluster_rmse_list = []
    all_mse_list = []
    mse_per_cluster_list = []
    
    for chosen_cluster_id in range(0, clusters_num):  # -1 means all cluster ids. cluster id 0 was not clustered by Tzu Hsi but I still use it
        print('Cluster ID: ', chosen_cluster_id)
        
        # -------------- Data Preprocessing --------------
        cluster_cases_df, cluster_vel_cases_df, proc_population_df = preprocess_dataset(cases_df.copy(), vel_cases_df.copy(), population_df.copy(), clusters, initial_date, forecast_final_date, chosen_cluster_id)
        localized_mean_vel_cases_series, future_mean_vel_cases_series, future_vel_cases_df, date_begin_sim, num_days_sim = process_date_data(cluster_vel_cases_df, initial_date=f'{initial_date}/2020', final_date=f'{cluster_final_date}/2020', num_days_future=num_days_future)

        # ------------- Create save folders --------------
        cluster_save_path = root_save_path + f'/cluster_{chosen_cluster_id}/'
        if os.path.isdir(cluster_save_path) is False:
            os.mkdir(cluster_save_path)

        # ---------- Forecast visualization------------
        visualize_trend_with_r_not(chosen_cluster_id, cluster_vel_cases_df, cluster_save_path)
        change_points = get_change_points(final_date=f'{cluster_final_date}/2020', final_change_date=final_change_date, cluster_id=chosen_cluster_id)
        Bayesian_Inference_SEIR.run(localized_mean_vel_cases_series, future_mean_vel_cases_series, proc_population_df.mean(axis=0), chosen_cluster_id,
                                   date_begin_sim, num_days_sim, cluster_save_path=cluster_save_path, change_points=change_points, N_SAMPLES=10000)

        # ---------- evaluation per county -----------
        cluster_mse_dict = eval_per_county(future_vel_cases_df, num_days_future, cluster_save_path)
        all_mse_list = all_mse_list + list(cluster_mse_dict.values())
        cluster_rmse_list.append(math.sqrt(statistics.mean(cluster_mse_dict.values())))
        print('cluster_mse_dict_values: ', len(cluster_mse_dict.values()))
        mse_per_cluster_list.append(list(cluster_mse_dict.values()))

    return all_mse_list, cluster_rmse_list, mse_per_cluster_list


if __name__ == "__main__":
    # Fitting the SEIR model to the data and estimating the parameters with the cluster id.

    forecast_final_date = '8/12'
    num_days_future = 14
    cluster_type = "no_constants"

    # initial_date_list = ['3/15', '4/1', '4/15', '5/1', '5/15']
    # cluster_final_date_list = ['4/30', '5/15', '5/31', '6/15', '6/30']
    # final_change_date_list = [datetime.datetime(2020, 4, 15), datetime.datetime(2020, 4, 30), datetime.datetime(2020, 5, 15), datetime.datetime(2020, 5, 31), datetime.datetime(2020, 6, 15)]

    initial_date_list = ['5/1']
    cluster_final_date_list = ['6/15']
    final_change_date_list = [datetime.datetime(2020, 5, 31)]
    for initial_date, cluster_final_date, final_change_date in zip(initial_date_list,  cluster_final_date_list, final_change_date_list):
        # load data
        initial_clusters = getTzuHsiClusters(column_date=f"{initial_date}~{cluster_final_date}", cluster_type=cluster_type)
        clusters_num = len(initial_clusters.unique())
        cases_df = pd.read_csv(f'../../generated/us_cases_counties.csv', header=0, index_col=0)
        vel_cases_df = pd.read_csv(f'../../generated/us_velocity_cases_counties.csv', header=0, index_col=0)
        population_df = johnsHopkinsPopulation()
        # population_df = pd.read_csv(f'../../generated/us_population_counties.csv', header=0, index_col=0)
        
        # set save path
        date_info = f'{initial_date.replace("/","-")}_{cluster_final_date.replace("/","-")}_{forecast_final_date.replace("/","-")}_{final_change_date.strftime("%m-%d")}_{cluster_type}'
        root_save_path = f'../../generated/plots/{date_info}/'
        if os.path.isdir(root_save_path) is False:
            os.mkdir(root_save_path)    
        
        # initial Parameters
        reassigned_clusters = initial_clusters
        REASSIGN_COUNTER_MAX = 5
        # load cluster data from intermediate reassign num
        reassign_counter_init = 0
        # root_save_path = f'../../generated/plots/{date_info}/reassign_{reassign_counter_init-1}/'
        # reassigned_clusters = pd.read_csv(root_save_path + f'clusters.csv', header=0, index_col=0)
        # reassigned_clusters = reassigned_clusters.iloc[:, 0]
        # reassigned_clusters = reassign_county(reassigned_clusters, clusters_num, cases_df, vel_cases_df, population_df, initial_date, cluster_final_date, forecast_final_date, num_days_future, root_save_path)
        
        # visualize clusters histogram
        # for reassign_counter in range(reassign_counter_init, REASSIGN_COUNTER_MAX):
            # root_save_path = f'../../generated/plots/{date_info}/reassign_{reassign_counter}/'
            # reassigned_clusters = pd.read_csv(root_save_path + f'clusters.csv', header=0, index_col=0)
            # reassigned_clusters = reassigned_clusters.iloc[:, 0]
            # histogram_clusters(reassigned_clusters, root_save_path)
            
        # Run Cycle Model
        for reassign_counter in range(reassign_counter_init, REASSIGN_COUNTER_MAX):
            root_save_path = f'../../generated/plots/{date_info}/reassign_{reassign_counter}/'
            if os.path.isdir(root_save_path) is False:
                os.mkdir(root_save_path)    
            reassigned_clusters.to_csv(root_save_path + f'clusters.csv')  # save current county assignment to clusters
        
            # ------------ Forecast -----------------
            all_mse_list, cluster_rmse_list, mse_per_cluster_list = forecast_main(reassigned_clusters, cases_df, vel_cases_df, population_df, clusters_num, initial_date, cluster_final_date, final_change_date_list, date_info, num_days_future, forecast_final_date, root_save_path)
            
            # ------------ Evaluation ---------------
            print('RMSE num:', len(all_mse_list))

            # --- for clustered dataset
            violin_mse_clusters(mse_per_cluster_list, root_save_path)
            average_of_rmse = round(math.sqrt(statistics.mean(all_mse_list)), 3)
            bar_rmse_clusters(clusters_num, cluster_rmse_list, 'RMSE of counties clustered with',
                              cluster_type, root_save_path, 'rmse_bar', average_of_rmse)

            # --- for unclustered dataset
            # unclustered_rmse_list = []
            # for chosen_cluster_id in range(0, clusters_num):
            #     print(chosen_cluster_id)
            #     local_mse_list = []
            #     chosen_cluster_series = reassigned_clusters[reassigned_clusters == chosen_cluster_id]
            #     cluster_counties = chosen_cluster_series.index.tolist()
            #     for a_county in cluster_counties:
            #         if a_county in cluster_mse_dict:
            #             local_mse_list.append(cluster_mse_dict[a_county])
            #     unclustered_rmse_list.append(math.sqrt(statistics.mean(local_mse_list)))
            # print(unclustered_rmse_list)
            #
            # average_of_rmse = round(statistics.mean(unclustered_rmse_list),3)
            # bar_rmse_clusters(clusters_num, unclustered_rmse_list, 'RMSE of unclustered counties', cluster_type, date_info, 'unclustered_rmse_bar', average_of_rmse)
            
            # ------------------------------
            reassigned_clusters = reassign_county(reassigned_clusters, clusters_num, cases_df, vel_cases_df, population_df, initial_date, cluster_final_date, forecast_final_date, num_days_future, root_save_path)
            

