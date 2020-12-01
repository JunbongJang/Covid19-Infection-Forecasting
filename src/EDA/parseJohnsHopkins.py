'''
Author: Junbong Jang
Date 6/18/2020

Parses Johns Hopkins Covid-19 Data into Velocity data
'''
import urllib.request
import pandas as pd
from src.EDA.parseData import state_to_abbr


def johnsHopkinsPopulation():
    population_df = pd.read_csv(f'../../assets/us_population_counties_jh.csv', header=0, index_col=False)
    population_df['State'] = population_df['State'].apply(state_to_abbr)
    population_df['state_county'] = population_df[['State', 'County']].agg('_'.join, axis=1)
    population_df['state_county'] = population_df['state_county'].str.lower()
    population_df = population_df.set_index('state_county')
    population_series = population_df['Number']

    return population_series


def getTzuHsiClusters(column_date, cluster_type):
    # print('getTzuHsiClusters')
    clusters_df = pd.read_csv(f'../../assets/us_county_clusters_{cluster_type}.csv', header=0, index_col=False)
    clusters_df['State'] = clusters_df['State'].apply(state_to_abbr)
    clusters_df['state_county'] = clusters_df[['State', 'County']].agg('_'.join, axis=1)
    clusters_df['state_county'] = clusters_df['state_county'].str.lower()
    clusters_df = clusters_df.set_index('state_county')

    clusters_series = clusters_df[column_date]

    return clusters_series

