'''
Author: Junbong Jang
Date 4/29/2020

Load timeseries data, train the model, and forecast

'''

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from datetime import date
import datetime
import os
import statistics 
import math

from src.EDA.parseJohnsHopkins import getTzuHsiClusters, johnsHopkinsPopulation
from src.Forecast import Bayesian_Inference_SEIR
from src.Forecast.Bayesian_Inference_SEIR_changepoints import get_change_points
from src.Forecast.timeseries_eval import *

def visualize_trend(ax, mean_cases_series, vel_cases_df, days, chosen_cluster, Rnot_index,
                          Rsquared, Rnot):
                         
    # plot lines for counties
    for a_col in vel_cases_df.columns:
        ax.plot(days, (vel_cases_df[a_col]/20).values.tolist(), linewidth=1) # divide by 10 for visualization
    
    # plot a line for mean values of all counties
    main_line = ax.plot(days, mean_cases_series.values.tolist(), linewidth=3)
    
    # xtick decoration 
    from matplotlib.ticker import (MultipleLocator, NullLocator)
    dates = vel_cases_df.index.tolist()
    
    spaced_out_dates=['']  # empty string elem needed because first element is skipped when using set_major_locator
    for date_index in range(0, len(dates)):
        dates[date_index] = dates[date_index].strftime("%m/%d/%y")
        if date_index%10 == 0:
            spaced_out_dates.append(str(dates[date_index]).replace('/2020',''))
        if str(dates[date_index]) in ['04/30/20', '05/31/20', '06/30/20', '07/31/20']:
            plt.axvline(x=date_index)
    plt.xticks(range(len(spaced_out_dates)), spaced_out_dates, size='small', rotation = (45))

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(NullLocator())

    text_input = ''  # r'$R^2$ =' + str(Rsquared) + '\n' + r'$R_0$ = ' + str(Rnot)
    ax.text(0.12, 0.77, text_input, color='black',
            bbox=dict(facecolor='none', edgecolor='black', pad=10.0),
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax.transAxes)

    # plt.axvline(x=Rnot_index, color='k', linestyle='dotted')

    plt.title(f'Reported COVID-19 Case Trend in Cluster {chosen_cluster}', fontsize='x-large')
    plt.xlabel('Dates', fontsize='large')
    plt.ylabel('New Confirmed\nCases in each day', fontsize='large')

    return main_line


def find_Rnot_index(mean_cases_series):
    Rnot_max_index = np.argmax(mean_cases_series) + 1
    temp_max_cases = 0
    temp_max_counter = 0
    tolerance = 0.01
    for Rnot_index, one_cases in enumerate(mean_cases_series):
        temp_max_counter = temp_max_counter + 1
        if one_cases > temp_max_cases + tolerance:
            temp_max_cases = one_cases
            temp_max_counter = 0
        if temp_max_counter > 10:
            Rnot_index = Rnot_index - 10
            break
    if Rnot_index > Rnot_max_index:
        Rnot_index = Rnot_max_index
    return Rnot_index


def visualize_trend_with_r_not(chosen_cluster_id, cluster_vel_cases_df, root_save_path):
    # min-max normalize data
    for a_column in cluster_vel_cases_df.columns:
        cluster_vel_cases_df[a_column] = (cluster_vel_cases_df[a_column] - cluster_vel_cases_df[a_column].min()) / (cluster_vel_cases_df[a_column].max() - cluster_vel_cases_df[a_column].min())
    
    mean_cluster_vel_cases_series = cluster_vel_cases_df.mean(axis=1)

    # process data by summation, log, shortening
    days = np.linspace(1, mean_cluster_vel_cases_series.size, mean_cluster_vel_cases_series.size)
    log_mean_cases_series = np.log(mean_cluster_vel_cases_series.values)

    # find Rnot_index
    '''
    Rnot_index = find_Rnot_index(mean_cluster_vel_cases_series)
    print('Rnot_index', Rnot_index)
    shortening using Rnot_index
    if np.any(np.isneginf(log_mean_cases_series)):
        score_first = -1
        coef_first = -1
    else:
        infectious_period = 6.6
        shortened_mean_summed_cases_series = log_mean_cases_series[:Rnot_index]
        shortened_days = days[:Rnot_index]
        score, coef, intercept = fit_R(shortened_days.reshape(-1, 1), shortened_mean_summed_cases_series.reshape(-1, 1))
        score_first = round(score, 3)
        coef_first = round(coef * infectious_period + 1, 3)
    '''
    score_first = 0
    coef_first = 0

    fig, ax = plt.subplots()
    main_line = visualize_trend(ax, mean_cluster_vel_cases_series, cluster_vel_cases_df, days, chosen_cluster_id,
                                      Rnot_index=0,
                                      Rsquared=score_first, Rnot=coef_first)

    # draw curve fit line
    '''
    if not np.any(np.isneginf(log_mean_cases_series)):
        shortened_mean_summed_cases_series = log_mean_cases_series[:Rnot_index]
        shortened_days = days[:Rnot_index]
        curve_fit = np.polyfit(shortened_days, shortened_mean_summed_cases_series, 1)
        y_days = np.exp(curve_fit[0] * shortened_days) * np.exp(curve_fit[1])
        curve_fit_line = ax.plot(shortened_days, y_days, '--')

        plt.legend((main_line[0], curve_fit_line[0]), ('Average of all counties', 'Curve Fit'),
                loc='upper left')
    else:
        plt.legend(main_line, ['Average of all counties'], loc='upper left')
    '''
    fig.tight_layout()
    fig.savefig(root_save_path + 'reported_cases_trend.png', dpi=fig.dpi)
    plt.close()


def bar_rmse_clusters(clusters_num, cluster_rmse_mean_list, title_prefix, cluster_type, date_info, save_name, average_of_rmse):
    ax = plt.gca()
    ax.bar(range(clusters_num),cluster_rmse_mean_list)
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('RMSE per cluster')
    ax.set_title(title_prefix + ' ' + cluster_type)
    ax.axhline(average_of_rmse, color='red')
    ax.text(0.97, 0.94, 'Average RMSE:%.3f' % (average_of_rmse), color='tab:red', horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
    ax.set_ylim(0.075, 0.275)
    
    plt.savefig(f'../../generated/plots/{date_info}/' + f'{save_name}.png')
    plt.close()



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

        clusters = getTzuHsiClusters(column_date=f"{initial_date}~{cluster_final_date}", cluster_type=cluster_type)
        clusters_num = len(clusters.unique())
        cluster_rmse_list = []
        all_mse_list = []
        mse_per_cluster_list = []

        for chosen_cluster_id in range(0, clusters_num):  # -1 means all cluster ids. cluster id 0 was not clustered by Tzu Hsi but I still use it
            print('Cluster ID: ', chosen_cluster_id)
            # load data
            # population_df = pd.read_csv(f'../../generated/us_population_counties.csv', header=0, index_col=0)
            population_df = johnsHopkinsPopulation()
            cases_df = pd.read_csv(f'../../generated/us_cases_counties.csv', header=0, index_col=0)
            vel_cases_df = pd.read_csv(f'../../generated/us_velocity_cases_counties.csv', header=0, index_col=0)

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
            print(cluster_vel_cases_df.shape, cluster_cases_df.shape)

            # -------------- Data Preprocessing --------------
            localized_mean_vel_cases_series, future_mean_vel_cases_series, future_vel_cases_df, date_begin_sim, num_days_sim = Bayesian_Inference_SEIR.process_date_data(cluster_vel_cases_df, initial_date=f'{initial_date}/2020', final_date=f'{cluster_final_date}/2020', num_days_future=num_days_future)

            # ------------- Create save folders --------------
            date_info = f'{initial_date.replace("/","-")}_{cluster_final_date.replace("/","-")}_{forecast_final_date.replace("/","-")}_{final_change_date.strftime("%m-%d")}_{cluster_type}'
            root_save_path = f'../../generated/plots/{date_info}'
            if os.path.isdir(root_save_path) is False:
                os.mkdir(root_save_path)
            root_save_path = root_save_path + f'/cluster_{chosen_cluster_id}/'
            if os.path.isdir(root_save_path) is False:
                os.mkdir(root_save_path)

            # ---------- Forecast visualization------------
            # visualize_trend_with_r_not(chosen_cluster_id, cluster_vel_cases_df, root_save_path)
            # change_points = get_change_points(final_date=f'{cluster_final_date}/2020', final_change_date=final_change_date, cluster_id=chosen_cluster_id)
            # Bayesian_Inference_SEIR.run(localized_mean_vel_cases_series, future_mean_vel_cases_series, population_df.mean(axis=0), chosen_cluster_id,
            #                            date_begin_sim, num_days_sim, root_save_path=root_save_path, change_points=change_points, N_SAMPLES=10000)

            # ---------- evaluation per county -----------

            cluster_cases_forecast = np.load(root_save_path + 'cases_forecast.npy')
            median_cluster_cases_forecast = np.median(cluster_cases_forecast, axis=-1)

            cluster_mse_dict = {}
            future_vel_cases_df = future_vel_cases_df.fillna(0)
            for county in future_vel_cases_df.columns:
                mse = mse_eval(future_vel_cases_df[county].tolist(), median_cluster_cases_forecast)

                cluster_mse_dict[county] = mse
                all_mse_list.append(mse)
                assert len(future_vel_cases_df[county].tolist()) == num_days_future
                assert len(future_vel_cases_df[county].tolist()) == len(median_cluster_cases_forecast)

            cluster_rmse_list.append(math.sqrt(statistics.mean(cluster_mse_dict.values())))
            print(len(cluster_mse_dict.values()))
            mse_per_cluster_list.append(list(cluster_mse_dict.values()))

        print('RMSE num:', len(all_mse_list))

        # for clustered dataset
        ax = sns.violinplot(data=mse_per_cluster_list)
        plt.title('MSE distribution per cluster')
        plt.xlabel('Cluster ID')
        plt.ylabel('MSE')
        plt.savefig(f'../../generated/plots/{date_info}/' + f'mse_violin.png')
        plt.close()

        average_of_rmse = round(math.sqrt(statistics.mean(all_mse_list)), 3)
        bar_rmse_clusters(clusters_num, cluster_rmse_list, 'RMSE of counties clustered with',
                          cluster_type, date_info, 'rmse_bar', average_of_rmse)

        #-------------------------------
        # for unclustered dataset only
        # unclustered_rmse_list = []
        # for chosen_cluster_id in range(0, clusters_num):
        #     print(chosen_cluster_id)
        #     local_mse_list = []
        #     chosen_cluster_series = clusters[clusters == chosen_cluster_id]
        #     cluster_counties = chosen_cluster_series.index.tolist()
        #     for a_county in cluster_counties:
        #         if a_county in cluster_mse_dict:
        #             local_mse_list.append(cluster_mse_dict[a_county])
        #     unclustered_rmse_list.append(math.sqrt(statistics.mean(local_mse_list)))
        # print(unclustered_rmse_list)
        #
        # average_of_rmse = round(statistics.mean(unclustered_rmse_list),3)
        # bar_rmse_clusters(clusters_num, unclustered_rmse_list, 'RMSE of unclustered counties', cluster_type, date_info, 'unclustered_rmse_bar', average_of_rmse)


