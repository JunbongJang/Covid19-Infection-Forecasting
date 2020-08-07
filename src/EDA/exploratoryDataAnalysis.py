'''
Author: Junbong Jang
Date 4/9/2020

Explores parsed NYTimes Covid-19 Data
'''

import pandas as pd
import seaborn as sns
from cycler import cycler
import matplotlib.pyplot as plt


def get_counties_with_n_cases(df, num_cases, days_before):
    last_row = df.iloc[-(1+days_before)]
    selected_county_list = []
    for state_county in df.columns.tolist():
        if last_row[state_county] > num_cases:
            selected_county_list.append(state_county)
    print(selected_county_list)
    print(len(selected_county_list))
    return selected_county_list


def rank_counties_by_cases(df):
    last_row = df.iloc[-1]
    last_row = last_row.sort_values(ascending=False)
    n_highest_cases = last_row[0:10]
    n_highest_cases = n_highest_cases.sort_values(ascending=True)
    ax = n_highest_cases.plot.barh(align='center', linewidth=3)

    plt.xlabel('Number of Cases', fontsize='large')
    plt.title(f'Ranking of Top 10 Counties', fontsize='x-large')
    plt.tight_layout()
    [i[1].set_linewidth(2) for i in ax.spines.items()] # make outline thicker
    plt.show()

    return n_highest_cases.index.tolist()


def plot_curve_per_county(df, state_county_list, y_label):
    days_cutoff_constant = 52 # first 49 days are ignored
    df = df.iloc[days_cutoff_constant:]
    line_array = []
    for state_county in state_county_list:
        state_county_series = df.loc[:,state_county]
        Days = range(len(state_county_series.index))
        Cases = state_county_series.values
        line_array.append(plt.plot(Days, Cases))

    plt.legend(state_county_list)
    plt.xlabel('Days', fontsize='large')
    plt.ylabel(y_label, fontsize='large')
    plt.title('Comparison of {} Progress per County'.format(y_label), fontsize='x-large')
    plt.tight_layout()
    plt.grid()
    plt.show()


def get_counties_with_rapid_increase(df):
    idxmax_series = df.idxmax(axis=0)
    max_velocity_series = pd.Series(0,  index=df.columns.tolist())

    for state_county, date in idxmax_series.iteritems():
        max_velocity_series[state_county] = df.loc[date, state_county]
    max_velocity_series = max_velocity_series.sort_values(ascending=False)
    n_highest_velocity = max_velocity_series[0:10]

    return n_highest_velocity.index.tolist()
    # n_highest_cases = last_row[0:10]
    # n_highest_cases = n_highest_cases.sort_values(ascending=True)
    # plt.title('Rank of Counties by Rapid Increase', fontsize='x-large')


if __name__ == "__main__":
    cases_data = pd.read_csv('../../generated/us_cases_counties.csv', header=0, index_col=0)
    velocity_data = pd.read_csv('../../generated/us_velocity_proc_cases_counties.csv', header=0, index_col=0)

    selected_county_list = get_counties_with_n_cases(cases_data, num_cases=-1, days_before=0)
    selected_county_list = get_counties_with_n_cases(cases_data, num_cases=10, days_before=0)
    selected_county_list = get_counties_with_n_cases(cases_data, num_cases=10, days_before=10)
    selected_county_list = get_counties_with_n_cases(cases_data, num_cases=10, days_before=20)
    selected_county_list = get_counties_with_n_cases(cases_data, num_cases=10, days_before=30)

    n_highest_cases = rank_counties_by_cases(cases_data)
    plot_curve_per_county(cases_data,n_highest_cases, y_label='Cases')
    # n_highest_velocity = get_counties_with_rapid_increase(velocity_data)
    plot_curve_per_county(velocity_data, n_highest_cases, y_label='Velocity')
