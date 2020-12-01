'''
Author: Junbong Jang
Date 4/9/2020

Explores and parses Covid-19 Data
'''

import pandas as pd
import seaborn as sns
from cycler import cycler
import matplotlib.pyplot as plt
import re


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


def correct_county_name_from_df(df):
    df['state_county'] = df['state_county'].str.lower()
    df['state_county'] = df['state_county'].str.replace(r' county$', '')
    df['state_county'] = df['state_county'].str.replace(r' county and city$', '')
    df['state_county'] = df['state_county'].str.replace(r' (including other portions of Kansas City)$', '')
    df['state_county'] = df['state_county'].str.replace(r' city and borough$', '')
    df['state_county'] = df['state_county'].str.replace(r'city and borough of', '')
    df['state_county'] = df['state_county'].str.replace(r' census area$', '')
    df['state_county'] = df['state_county'].str.replace(r' area$', '')
    df['state_county'] = df['state_county'].str.replace(r' parish$', '')
    df['state_county'] = df['state_county'].str.replace(r' borough$', '')
    df['state_county'] = df['state_county'].str.replace(r' municipality$', '')
    df['state_county'] = df['state_county'].str.replace(r'municipality of ', '')
    df['state_county'] = df['state_county'].str.replace(r'dc_washington', 'dc_district of columbia')

    skip_list = ['va_fairfax city', 'va_franklin city', 'va_richmond city', 'va_roanoke city', 'md_baltimore city', 'mo_st. louis city', 'va_charles city', 'va_james city', 'nv_carson city']
    for index, row in df.iterrows():
        if row['state_county'] in skip_list:
            continue
            # print(df.at[index, 'state_county'], 'skipped')
        else:
            df.at[index, 'state_county'] = re.sub(r' city$', '', df.at[index, 'state_county'])

    # if re.match(regex, content) is not None:
    # df['state_county'] = df['state_county'].str.replace(r'va_fairfax', ')
    # df['state_county'] = df['state_county'].str.replace(r'va_franklin', )
    # df['state_county'] = df['state_county'].str.replace(r'va_richmond', ')
    # df['state_county'] = df['state_county'].str.replace(r'va_roanoke', )
    # df['state_county'] = df['state_county'].str.replace(r'md_baltimore', )
    # df['state_county'] = df['state_county'].str.replace(r'mo_st. louis', )
    # df['state_county'] = df['state_county'].str.replace(r'va_charles', )
    # df['state_county'] = df['state_county'].str.replace(r'va_james', )
    # df['state_county'] = df['state_county'].str.replace(r'nv_carson', )

    return df

def state_to_abbr(a_state):
    state_dict = {
        'Alabama': {
            'state_cd': '01',
            'state_abbr': 'AL'
        },
        'Alaska': {
            'state_cd': '02',
            'state_abbr': 'AK'
        },
        'Arizona': {
            'state_cd': '04',
            'state_abbr': 'AZ'
        },
        'Arkansas': {
            'state_cd': '05',
            'state_abbr': 'AR'
        },
        'California': {
            'state_cd': '06',
            'state_abbr': 'CA'
        },
        'Colorado': {
            'state_cd': '08',
            'state_abbr': 'CO'
        },
        'Connecticut': {
            'state_cd': '09',
            'state_abbr': 'CT'
        },
        'Delaware': {
            'state_cd': '10',
            'state_abbr': 'DE'
        },
        'District of Columbia': {
            'state_cd': '11',
            'state_abbr': 'DC'
        },
        'Florida': {
            'state_cd': '12',
            'state_abbr': 'FL'
        },
        'Georgia': {
            'state_cd': '13',
            'state_abbr': 'GA'
        },
        'Hawaii': {
            'state_cd': '15',
            'state_abbr': 'HI'
        },
        'Idaho': {
            'state_cd': '16',
            'state_abbr': 'ID'
        },
        'Illinois': {
            'state_cd': '17',
            'state_abbr': 'IL'
        },
        'Indiana': {
            'state_cd': '18',
            'state_abbr': 'IN'
        },
        'Iowa': {
            'state_cd': '19',
            'state_abbr': 'IA'
        },
        'Kansas': {
            'state_cd': '20',
            'state_abbr': 'KS'
        },
        'Kentucky': {
            'state_cd': '21',
            'state_abbr': 'KY'
        },
        'Louisiana': {
            'state_cd': '22',
            'state_abbr': 'LA'
        },
        'Maine': {
            'state_cd': '23',
            'state_abbr': 'ME'
        },
        'Maryland': {
            'state_cd': '24',
            'state_abbr': 'MD'
        },
        'Massachusetts': {
            'state_cd': '25',
            'state_abbr': 'MA'
        },
        'Michigan': {
            'state_cd': '26',
            'state_abbr': 'MI'
        },
        'Minnesota': {
            'state_cd': '27',
            'state_abbr': 'MN'
        },
        'Mississippi': {
            'state_cd': '28',
            'state_abbr': 'MS'
        },
        'Missouri': {
            'state_cd': '29',
            'state_abbr': 'MO'
        },
        'Montana': {
            'state_cd': '30',
            'state_abbr': 'MT'
        },
        'Nebraska': {
            'state_cd': '31',
            'state_abbr': 'NE'
        },
        'Nevada': {
            'state_cd': '32',
            'state_abbr': 'NV'
        },
        'New Hampshire': {
            'state_cd': '33',
            'state_abbr': 'NH'
        },
        'New Jersey': {
            'state_cd': '34',
            'state_abbr': 'NJ'
        },
        'New Mexico': {
            'state_cd': '35',
            'state_abbr': 'NM'
        },
        'New York': {
            'state_cd': '36',
            'state_abbr': 'NY'
        },
        'North Carolina': {
            'state_cd': '37',
            'state_abbr': 'NC'
        },
        'North Dakota': {
            'state_cd': '38',
            'state_abbr': 'ND'
        },
        'Ohio': {
            'state_cd': '39',
            'state_abbr': 'OH'
        },
        'Oklahoma': {
            'state_cd': '40',
            'state_abbr': 'OK'
        },
        'Oregon': {
            'state_cd': '41',
            'state_abbr': 'OR'
        },
        'Pennsylvania': {
            'state_cd': '42',
            'state_abbr': 'PA'
        },
        'Rhode Island': {
            'state_cd': '44',
            'state_abbr': 'RI'
        },
        'South Carolina': {
            'state_cd': '45',
            'state_abbr': 'SC'
        },
        'South Dakota': {
            'state_cd': '46',
            'state_abbr': 'SD'
        },
        'Tennessee': {
            'state_cd': '47',
            'state_abbr': 'TN'
        },
        'Texas': {
            'state_cd': '48',
            'state_abbr': 'TX'
        },
        'Utah': {
            'state_cd': '49',
            'state_abbr': 'UT'
        },
        'Vermont': {
            'state_cd': '50',
            'state_abbr': 'VT'
        },
        'Virginia': {
            'state_cd': '51',
            'state_abbr': 'VA'
        },
        'Washington': {
            'state_cd': '53',
            'state_abbr': 'WA'
        },
        'West Virginia': {
            'state_cd': '54',
            'state_abbr': 'WV'
        },
        'Wisconsin': {
            'state_cd': '55',
            'state_abbr': 'WI'
        },
        'Wyoming': {
            'state_cd': '56',
            'state_abbr': 'WY'
        },
        'American Samoa': {
            'state_cd': '60',
            'state_abbr': 'AS'
        },
        'Federated States of Micronesia': {
            'state_cd': '64',
            'state_abbr': 'FM     '
        },
        'Guam     ': {
            'state_cd': '66',
            'state_abbr': 'GU'
        },
        'Marshall Islands': {
            'state_cd': '68',
            'state_abbr': 'MH'
        },
        'Commonwealth of the Northern Mariana Islands': {
            'state_cd': '69',
            'state_abbr': 'MP'
        },
        'Palau': {
            'state_cd': '70',
            'state_abbr': 'PW'
        },
        'Puerto Rico': {
            'state_cd': '72',
            'state_abbr': 'PR'
        },
        'U.S. Minor Outlying Islands': {
            'state_cd': '74',
            'state_abbr': 'UM'
        },
        'U.S. Virgin Islands': {
            'state_cd': '78',
            'state_abbr': 'VI'
        }
    }

    return state_dict[a_state]['state_abbr']

if __name__ == "__main__":
    cases_data = pd.read_csv('../../generated/us_cases_counties_0531.csv', header=0, index_col=0)
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
