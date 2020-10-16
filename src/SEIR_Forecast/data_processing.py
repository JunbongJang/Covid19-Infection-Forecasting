
import datetime

def preprocess_dataset(cases_df, vel_cases_df, population_df, clusters, initial_date, forecast_final_date, chosen_cluster_id):
    # change index which are date string to datetime objects
    new_index_dict = {}
    for an_index in vel_cases_df.index:
        new_index_dict[an_index] = datetime.datetime.strptime(an_index, '%m/%d/%Y')
    vel_cases_df = vel_cases_df.rename(new_index_dict, axis='index')

    if chosen_cluster_id == -1:
        chosen_cluster_id = 'All'
        chosen_cluster_series = clusters
    else:
        chosen_cluster_series = clusters[clusters == chosen_cluster_id]
    cluster_counties = chosen_cluster_series.index.tolist()

    # get counties in current cluster only
    cleaned_cluster_counties = []
    for a_county in cluster_counties:
        if a_county in vel_cases_df.columns.to_numpy() and a_county in population_df.index.to_numpy():
            if vel_cases_df[a_county].isnull().values.any():
                print('error')
                exit()
            cleaned_cluster_counties.append(a_county)
        else:
            continue
    print('Cluster Counties: ', len(cluster_counties), len(cleaned_cluster_counties))

    # Get timeseries within a range, which Tzu-Hsi used for clustering
    population_df = population_df.loc[cleaned_cluster_counties]
    cluster_vel_cases_df = vel_cases_df[cleaned_cluster_counties]
    cluster_vel_cases_df = cluster_vel_cases_df.loc[f'{initial_date}/2020':f'{forecast_final_date}/2020']
    cluster_cases_df = cases_df[cleaned_cluster_counties]
    cluster_cases_df = cluster_cases_df.loc[f'{initial_date}/2020':f'{forecast_final_date}/2020']
    print('cluster cases df shape: ', cluster_vel_cases_df.shape, cluster_cases_df.shape)
    
    return cluster_cases_df, cluster_vel_cases_df, population_df


def process_date_data(cluster_vel_cases_df, initial_date, final_date, num_days_future):
    diff_data_sim = 0
    date_data_begin = datetime.datetime.strptime(initial_date, '%m/%d/%Y')
    date_data_end = datetime.datetime.strptime(final_date, '%m/%d/%Y')
    
    date_begin_sim = date_data_begin - datetime.timedelta(days = diff_data_sim)
    date_end_sim   = date_data_end   + datetime.timedelta(days = num_days_future) 
    
    num_days_sim = (date_end_sim-date_begin_sim + datetime.timedelta(days=1)).days
    print(date_begin_sim)
    print(date_end_sim)
    print(date_data_begin)
    print(date_data_end)
    # min-max normalize data
    for a_column in cluster_vel_cases_df.columns:
        cluster_vel_cases_df[a_column] = (cluster_vel_cases_df[a_column] - cluster_vel_cases_df[a_column].min()) / (cluster_vel_cases_df[a_column].max() - cluster_vel_cases_df[a_column].min())
    localized_vel_cases_df = cluster_vel_cases_df.loc[date_data_begin: date_data_end]
    future_vel_cases_df = cluster_vel_cases_df.loc[date_data_end + datetime.timedelta(days = 1): date_end_sim]
    print(localized_vel_cases_df.shape)
    print(future_vel_cases_df.shape)
    
    localized_mean_vel_cases_series = localized_vel_cases_df.mean(axis=1)
    future_mean_vel_cases_series = future_vel_cases_df.mean(axis=1)

    return localized_mean_vel_cases_series, future_mean_vel_cases_series, future_vel_cases_df, date_begin_sim, num_days_sim
    