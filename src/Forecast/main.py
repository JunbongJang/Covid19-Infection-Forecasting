'''
Author: Junbong Jang
Date 4/29/2020

Load timeseries data, train the model, and forecast

'''

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from datetime import date
import datetime
import os

from src.EDA.parseJohnsHopkins import getTzuHsiCluster, johnsHopkinsPopulation
from src.Forecast.Bayesian_Inference_SEIR import run_bayesian_inference_SEIR


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
        if date_index%10 == 0:
            spaced_out_dates.append(str(dates[date_index]).replace('/2020',''))
        if str(dates[date_index]) in ['4/30/2020', '5/31/2020', '6/30/2020', '7/31/2020']:
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


if __name__ == "__main__":
    # Fitting the SEIR model to the data and estimating the parameters with the cluster id.

    # 3/15~4/30	3/15~5/15	3/15~5/31	3/15~6/15	3/15~6/30	3/15~7/15	3/15~7/31
    initial_date = '3/15'
    cluster_final_date = '4/30'
    forecast_final_date = '8/12'
    
    for chosen_cluster_id in range(0,5):  # 0 means all cluster ids
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
        
        print('Cluster ID: ', chosen_cluster_id)
        cluster_counties = getTzuHsiCluster(chosen_cluster_id, column_date=f"{initial_date}~{cluster_final_date}")

        # get counties in current cluster only
        cleaned_cluster_counties = []
        for a_county in cluster_counties:
            if a_county in vel_cases_df.columns.to_numpy() and a_county in population_df.index.to_numpy():
                cleaned_cluster_counties.append(a_county)
            else:
                #print(a_county + ' skipped')
                continue
        print('Cluster Counties: ', len(cluster_counties), len(cleaned_cluster_counties))

        # Get timeseries within a range, which Tzu-Hsi used for clustering
        population_df = population_df.loc[cleaned_cluster_counties]
        cluster_vel_cases_df = vel_cases_df[cleaned_cluster_counties]
        cluster_vel_cases_df = cluster_vel_cases_df.loc[f'{initial_date}/2020':f'{forecast_final_date}/2020']
        cluster_cases_df = cases_df[cleaned_cluster_counties]
        cluster_cases_df = cluster_cases_df.loc[f'{initial_date}/2020':f'{forecast_final_date}/2020']
        print(cluster_vel_cases_df.shape, cluster_cases_df.shape)

        # --- plot save path ----
        root_save_path = f'../../generated/plots/cluster_{chosen_cluster_id}/'
        if os.path.isdir(root_save_path) is False:
            os.mkdir(root_save_path)

        # visualize_trend_with_r_not(chosen_cluster_id, cluster_vel_cases_df, root_save_path)
        
        run_bayesian_inference_SEIR(cluster_vel_cases_df, population_df.mean(axis=0), chosen_cluster_id, 
                                   initial_date=f'{initial_date}/2020', final_date=f'{cluster_final_date}/2020', root_save_path=root_save_path, N_SAMPLES=10000)


