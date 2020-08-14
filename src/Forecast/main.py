'''
Author: Junbong Jang
Date 4/29/2020

Loads time-series data, train the model with them, and make predictions
'''

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from datetime import date
from pandas.plotting import autocorrelation_plot

from src.EDA.parseJohnsHopkins import getTzuHsiCluster, johnsHopkinsPopulation
from src.Forecast.Bayesian_MCMC import *
from src.Forecast.SEIR import SEIR, SEIR_actual_graph, fit_R
from src.Forecast.other_models import *
from src.Forecast.Bayesian_Inference_SEIR import run_bayesian_inference_SEIR

def split_data(df):
    # split into train and test sets
    X = df.values
    train_size = int(len(X) * 0.7)
    train, test = X[1:train_size], X[train_size:]
    train_X, train_y = train[:,0], train[:,1]  # date, value
    test_X, test_y = test[:,0], test[:,1]  # date, value
    return train_X, train_y, test_X, test_y


def visualize_predictions(file_type, chosen_county, train_y, test_y, predictions):
    # plot predictions and expected results
    line1, = plt.plot(train_y)
    line2, = plt.plot([None for i in train_y] + [x for x in test_y])
    line3, = plt.plot([None for i in train_y] + [x for x in predictions])

    plt.legend((line1, line2, line3), ('Training Data', 'Ground Truth', 'Prediction'))
    plt.xlabel('Days', fontsize='large')
    plt.ylabel(f'Velocity of {file_type}', fontsize='large')
    plt.title(f'Forecast velocity of {file_type} in {chosen_county}', fontsize='x-large')

    plt.show()


def visualize_SEIR_predictions(file_type, chosen_county, test_y, predictions):
    # plot predictions and expected results
    line2, = plt.plot([x for x in test_y])
    line3, = plt.plot([x for x in predictions])

    plt.legend((line2, line3), ('Ground Truth', 'Prediction'))
    plt.xlabel('Days', fontsize='large')
    plt.ylabel(f'{file_type}', fontsize='large')
    plt.title(f'Forecast of {file_type} in {chosen_county}', fontsize='x-large')

    plt.show()


def visualize_input(df, chosen_county):
    X = df.values
    autocorrelation_plot(X[:,1])
    plt.title(f'Autocorrelation at {chosen_county}', fontsize='x-large')
    plt.show()

    plt.plot(X[:, 1])
    plt.title(f'Case Velocity at {chosen_county}', fontsize='x-large')
    plt.xlabel('Days', fontsize='large')
    plt.ylabel('Case Velocity', fontsize='large')
    plt.show()


def visualize_cases_trend(ax, mean_cases_series, vel_cases_df, days, chosen_cluster, Rnot_index,
                          Rsquared, Rnot):
    # plot lines for counties
    for a_col in vel_cases_df.columns:
        ax.plot(days, (vel_cases_df[a_col]/10).values.tolist(), linewidth=1) # divide by 10 for visualization
    # plot one line
    main_line = ax.plot(days, mean_cases_series.values.tolist(), linewidth=3)

    # text_input = r'$R^2$ =' + str(Rsquared) + '\n' + r'$R_0$ = ' + str(Rnot)
    text_input = ''
    ax.text(0.12, 0.77, text_input, color='black',
            bbox=dict(facecolor='none', edgecolor='black', pad=10.0),
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax.transAxes)

    # plt.axvline(x=Rnot_index, color='k', linestyle='dotted')

    plt.title(f'Infection Trend of Cluster {chosen_cluster}', fontsize='x-large')
    plt.xlabel('Days since 3/15/20', fontsize='large')
    plt.ylabel('Change in Cases', fontsize='large')

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


def find_r_not(chosen_cluster_id, cluster_vel_cases_df, mean_cases_series):
    # process data by summation, log, shortening
    days = np.linspace(1, mean_cases_series.size, mean_cases_series.size)
    log_mean_cases_series = np.log(mean_cases_series.values)

    # find Rnot_index
    # Rnot_index = find_Rnot_index(mean_cases_series)
    # print('Rnot_index', Rnot_index)
    # shortening using Rnot_index
    if np.any(np.isneginf(log_mean_cases_series)):
        score_first = -1
        coef_first = -1
    # else:
    #     infectious_period = 6.6
    #     shortened_mean_summed_cases_series = log_mean_cases_series[:Rnot_index]
    #     shortened_days = days[:Rnot_index]
    #     score, coef, intercept = fit_R(shortened_days.reshape(-1, 1), shortened_mean_summed_cases_series.reshape(-1, 1))
    #     score_first = round(score, 3)
    #     coef_first = round(coef * infectious_period + 1, 3)

    score_first = 0
    coef_first = 0

    fig, ax = plt.subplots()
    main_line = visualize_cases_trend(ax, mean_cases_series, cluster_vel_cases_df, days, chosen_cluster_id,
                                      Rnot_index=0,
                                      Rsquared=score_first, Rnot=coef_first)

    # draw curve fit line
    # if not np.any(np.isneginf(log_mean_cases_series)):
    #     shortened_mean_summed_cases_series = log_mean_cases_series[:Rnot_index]
    #     shortened_days = days[:Rnot_index]
    #     curve_fit = np.polyfit(shortened_days, shortened_mean_summed_cases_series, 1)
    #     y_days = np.exp(curve_fit[0] * shortened_days) * np.exp(curve_fit[1])
    #     curve_fit_line = ax.plot(shortened_days, y_days, '--')
    #
    #     plt.legend((main_line[0], curve_fit_line[0]), ('Average of all counties', 'Curve Fit'),
    #                loc='upper left')
    # else:
    #     plt.legend(main_line, ['Average of all counties'], loc='upper left')

    fig.savefig(f'../../generated/plots/Cluster {chosen_cluster_id}.png', dpi=fig.dpi)
    plt.close()


if __name__ == "__main__":
    # 6/18/2020
    # Fitting the SEIR model to the data and estimating the parameters with the cluster id.

    for chosen_cluster_id in range(8,14):
        # load data
        # population_df = pd.read_csv(f'../../generated/us_population_counties.csv', header=0, index_col=0)
        population_df = johnsHopkinsPopulation()
        cases_df = pd.read_csv(f'../../generated/us_cases_counties_0531.csv', header=0, index_col=0)
        vel_cases_df = pd.read_csv(f'../../generated/us_velocity_cases_counties.csv', header=0, index_col=0)
        print(population_df)

        print('Cluster ID: ', chosen_cluster_id)
        cluster_counties = getTzuHsiCluster(chosen_cluster_id)

        # get counties in current cluster only
        cleaned_cluster_counties = []
        for a_county in cluster_counties:
            if a_county in vel_cases_df.columns.to_numpy() and a_county in population_df.index.to_numpy():
                cleaned_cluster_counties.append(a_county)
            else:
                print(a_county + ' skipped')
        print('Cluster Counties: ', len(cluster_counties), len(cleaned_cluster_counties))

        # Tzu-Hsi used that range of data
        population_df = population_df.loc[cleaned_cluster_counties]
        cluster_vel_cases_df = vel_cases_df[cleaned_cluster_counties]
        cluster_vel_cases_df = cluster_vel_cases_df.loc['3/15/20':'5/31/20']
        cluster_cases_df = cases_df[cleaned_cluster_counties]
        # cluster_cases_df = cluster_cases_df.loc['3/15/2020':'4/30/2020']
        print(cluster_vel_cases_df.shape)
        print(cluster_cases_df.shape)

        # min-max normalize data
        # for a_column in cluster_vel_cases_df.columns:
        #    cluster_vel_cases_df[a_column] = (cluster_vel_cases_df[a_column] - cluster_vel_cases_df[a_column].min()) / (cluster_vel_cases_df[a_column].max() - cluster_vel_cases_df[a_column].min())
        #
        cluster_vel_cases_series = cluster_vel_cases_df.mean(axis=1)
        date_array = cluster_vel_cases_series.index.to_numpy()
        #
        # find_r_not(chosen_cluster_id, cluster_vel_cases_df, cluster_vel_cases_series)


        mean_cases_series = cluster_cases_df.mean(axis=1)
        cluster_mean_population = population_df.mean(axis=0)
        mean_cases_series = mean_cases_series.reset_index(drop=True) # remove dates to a day

        cluster_vel_cases_series = cluster_vel_cases_series.to_numpy()
        cluster_cases_array = mean_cases_series.to_numpy()
        day_array = mean_cases_series.index.to_numpy()
        N_SAMPLES = 10000
        
        
        # get_time_variant_R(day_array, cluster_cases_array, cluster_mean_population, N_SAMPLES)
        # bayesian_inference_SEIR(day_array, cluster_cases_array, N_SAMPLES)
        run_bayesian_inference_SEIR(date_array, cluster_vel_cases_df, cluster_mean_population, chosen_cluster_id, N_SAMPLES)
        break

    '''
    # 4/20/2020
    
    # get data
    population_df = pd.read_csv(f'../../generated/us_population_counties.csv', header=0, index_col=0)
    deaths_df = pd.read_csv(f'../../generated/us_deaths_counties.csv', header=0, index_col=0)
    deaths_df = deaths_df.reset_index()
    cases_df = pd.read_csv(f'../../generated/us_cases_counties.csv', header=0, index_col=0)
    cases_df = cases_df.reset_index()
    velocity_cases_df = pd.read_csv(f'../../generated/us_velocity_proc_cases_counties.csv', header=0, index_col=0)
    velocity_cases_df = velocity_cases_df.reset_index()
    recovered_df = pd.read_csv(f'../../generated/us_recovered_states.csv', header=0, index_col=0)
    recovered_df = recovered_df.fillna(0)

    # pick one county
    chosen_county = 'NY_Nassau County'
    chosen_state = 'NY'

    # get lockdown day
    start_date = date(2020, 1, 22)
    lockdown_date = date(2020, 3, 22)
    delta = lockdown_date - start_date
    lockdown_day = delta.days

    # get date with the first case

    # get data for county from the first case

    chosen_population = population_df[chosen_county].values[0]
    chosen_deaths_df = deaths_df[['index', chosen_county]]
    chosen_cases_df = cases_df[['index', chosen_county]]
    chosen_vel_df = velocity_cases_df[['index', chosen_county]]
    chosen_recovered_df = recovered_df[[chosen_state]]/3
    # visualize_input(chosen_vel_df, chosen_county)

    # fit model
    predicted_s = SEIR(chosen_population, range(len(chosen_vel_df.values)), lockdown_day, chosen_county)
    actual_s = SEIR_actual_graph(chosen_population, range(len(chosen_cases_df.values)), chosen_cases_df[chosen_county].values.tolist(), chosen_recovered_df.values.tolist(), chosen_county)

    # evaluate model
    evaluate_predictions(actual_s, predicted_s)
    visualize_SEIR_predictions('cases', chosen_county, actual_s, predicted_s)

    # fit model
    # train_x, train_y, test_x, test_y = split_data(chosen_vel_df)
    # predictions = baseline_model(train_x, train_y, test_x, test_y)
    # predictions = ARIMA_model(train_y, test_y)
    # predictions = rf_model(train_y, test_y)

    # evaluate model
    # evaluate_predictions(test_y, predictions)
    # visualize_predictions('cases', chosen_county, train_y, test_y, predictions)
    '''
