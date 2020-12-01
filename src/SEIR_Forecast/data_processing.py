import datetime


def preprocess_dataset(cases_df, population_df, cluster_counties):
    # change index which are date string to datetime objects
    new_index_dict = {}
    for an_index in cases_df.index:
        new_index_dict[an_index] = datetime.datetime.strptime(an_index, '%m/%d/%y')
    cases_df = cases_df.rename(new_index_dict, axis='index')

    # get counties in current cluster only
    cleaned_cluster_counties = []
    for a_county in cluster_counties:
        if a_county in cases_df.columns.to_numpy() and a_county in population_df.index.to_numpy():
            if cases_df[a_county].isnull().values.any():
                print('error!!!')
                exit()
            cleaned_cluster_counties.append(a_county)
        else:
            print('skipped: ', a_county)
            continue
    print('Cluster Counties: ', len(cluster_counties), len(cleaned_cluster_counties))

    # Get data of counties within a Tzu-Hsi's cluster
    population_series = population_df.loc[cleaned_cluster_counties]
    cluster_cases_df = cases_df[cleaned_cluster_counties]
    print('preprocess_dataset: ', cluster_cases_df.shape)  # rows are dates, columns are counties

    return cluster_cases_df, population_series


def process_date(cluster_cases_df, initial_date, final_date, dataset_final_date, num_days_future):
    diff_data_sim = 0

    old_cumulative_infected_date = initial_date - datetime.timedelta(days=14)
    date_begin_sim = initial_date - datetime.timedelta(days = diff_data_sim)
    date_end_sim   = final_date   + datetime.timedelta(days = num_days_future)
    
    num_days_sim = (date_end_sim-date_begin_sim + datetime.timedelta(days=1)).days
    print(date_begin_sim, date_end_sim)
    print(initial_date, final_date)
    # min-max normalize data
    # for a_column in cluster_cases_df.columns:
    #     cluster_cases_df[a_column] = (cluster_cases_df[a_column] - cluster_cases_df[a_column].min()) / (cluster_cases_df[a_column].max() - cluster_cases_df[a_column].min())

    old_cumulative_infected_cases_series = cluster_cases_df.loc[old_cumulative_infected_date]
    cluster_cases_df = cluster_cases_df.loc[initial_date: dataset_final_date]
    current_cases_df = cluster_cases_df.loc[initial_date: final_date]
    future_cases_df = cluster_cases_df.loc[final_date + datetime.timedelta(days = 1): date_end_sim]
    print('cluster_cases_df:', cluster_cases_df.shape, 'current_cases_df:', current_cases_df.shape, 'future_cases_df:', future_cases_df.shape)

    return cluster_cases_df, current_cases_df, future_cases_df, old_cumulative_infected_cases_series, date_begin_sim, num_days_sim
    