from src.SEIR_Forecast.data_processing import *
from src.SEIR_Forecast.timeseries_eval import *


def reassign_county(original_clusters, clusters_num, cases_df, vel_cases_df, population_df, initial_date, final_date, num_days_future, root_save_path):
    '''
    Params:
        clusters
        time series forecast for each county
    
    find counties in each cluster group with high mse
    Assign those counties to another cluster group that yields the lowest mse
    
    Returns: 
        new counties per cluster
    '''
    print('reassign_county')
    reassigned_clusters = original_clusters.copy()

    for chosen_cluster_id in range(0, clusters_num):
        print()
        print('Cluster ID: ', chosen_cluster_id)
        
        # -------------- Data Preprocessing --------------
        if chosen_cluster_id == -1:
            chosen_cluster_id = 'All'
            chosen_cluster_series = clusters
        else:
            chosen_cluster_series = clusters[clusters == chosen_cluster_id]
        cluster_counties = chosen_cluster_series.index.tolist()


        cluster_cases_df, proc_population_series = preprocess_dataset(cases_df.copy(), population_df.copy(), cluster_counties)
        cluster_vel_cases_df, proc_population_series = preprocess_dataset(vel_cases_df.copy(), population_df.copy(), cluster_counties)

        cluster_cases_df, current_cumulative_cases_df, future_cumulative_cases_df, _, _, _ = \
            process_date(cluster_cases_df, initial_date, final_date, dataset_final_date, num_days_future)
        cluster_vel_cases_df, current_vel_cases_df, future_vel_cases_df, _, _, _ = \
            process_date(cluster_vel_cases_df, initial_date, final_date, dataset_final_date, num_days_future)

        # -------------------------------------------------
        # find a county from mse distribution
        cluster_save_path = root_save_path + f'/cluster_{chosen_cluster_id}/'
        cluster_mse_dict, cluster_rsquared_dict, cluster_mape_dict, cluster_wape_dict = eval_per_cluster(future_vel_cases_df, num_days_future, cluster_save_path)
        
        # outliers which are above the threshold are found
        sorted_cluster_mse_dict = {k: v for k, v in sorted(cluster_mse_dict.items(), key=lambda item: item[1])}
        sorted_mse_list = list(sorted_cluster_mse_dict.values())
        # upper_outlier_threshold = get_outlier_threshold_from_list(sorted_mse_list)
        # outlier_mse_dict = dict((k, v) for k, v in sorted_cluster_mse_dict.items() if v >= upper_outlier_threshold)
        
        # Each county is reassigned to the appropriate cluster group
        for county_name, old_mse in sorted_cluster_mse_dict.items():  # outlier_mse_dict.items()
            # find a cluster group that fit in well with this county
            print(county_name, old_mse)
            min_mse = old_mse  # default value
            new_cluster_id = chosen_cluster_id  # default cluster id
            for cluster_id in range(0, clusters_num):
                cluster_save_path = root_save_path + f'/cluster_{cluster_id}/'
                median_cluster_cases_forecast = get_median_cluster_cases_forecast(cluster_save_path)

                mse, r_squared, mape, wape = eval_a_county(future_vel_cases_df[county_name], median_cluster_cases_forecast)
                if min_mse > mse:
                    new_cluster_id = cluster_id
                    min_mse = mse
            
            print(new_cluster_id, min_mse)
            reassigned_clusters[county_name] = new_cluster_id
    
    return reassigned_clusters
        