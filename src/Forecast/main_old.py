'''
Author: Junbong Jang
Creation Date: 4/20/2020
'''

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from pandas.plotting import autocorrelation_plot

from src.Forecast.SEIR import SEIR, SEIR_actual_graph, fit_R
from src.Forecast.other_models import *

def split_data(df):
    # split into train and test sets
    X = df.values
    train_size = int(len(X) * 0.7)
    train, test = X[1:train_size], X[train_size:]
    train_X, train_y = train[:,0], train[:,1]  # date, value
    test_X, test_y = test[:,0], test[:,1]  # date, value
    return train_X, train_y, test_X, test_y


def visualize_SEIR_predictions(file_type, chosen_county, test_y, predictions):
    # plot predictions and expected results
    line2, = plt.plot([x for x in test_y])
    line3, = plt.plot([x for x in predictions])

    plt.legend((line2, line3), ('Ground Truth', 'Prediction'))
    plt.xlabel('Days', fontsize='large')
    plt.ylabel(f'{file_type}', fontsize='large')
    plt.title(f'Forecast of {file_type} in {chosen_county}', fontsize='x-large')

    plt.show()
    

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


if __name__ == "__main__":
    
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
    
    