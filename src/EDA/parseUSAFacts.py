'''
Author: Junbong Jang
Date 4/28/2020

Parses USAFacts Covid-19 Data into Velocity data
'''
import urllib.request
import pandas as pd
from parseData import correct_county_name_from_df


def download_raw_data(file_type):
    filename = f"../../assets/us_{file_type}_counties.csv"
    if file_type == 'cases':
        url = "https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_confirmed_usafacts.csv"
    elif file_type == 'deaths':
        url = "https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_deaths_usafacts.csv"
    else:
        url = "https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_county_population_usafacts.csv"
    filename, headers = urllib.request.urlretrieve(url, filename=filename)
    print("download file location: ", filename)
    print("download headers: ", headers)


def parseData(file_type):
    raw_df = pd.read_csv(f'../../assets/us_{file_type}_counties.csv', header=0, index_col=None)
    raw_df = raw_df.fillna(0)
    raw_df.sort_values(by=['State','County Name'])

    # create state_county column for loop optimization
    raw_df['state_county'] = raw_df[['State', 'County Name']].agg('_'.join, axis=1)
    raw_df['state_county'] = raw_df['state_county'].str.lower()
    raw_df = correct_county_name_from_df(raw_df)

    raw_df.sort_values(by=['state_county'])

    # remove statewide unallocated rows
    statewide_unalloced_indices = raw_df.index[raw_df['countyFIPS'] == 0]
    newyork_unalloced_indices = raw_df.index[raw_df['countyFIPS'] == 1]
    unalloced_indices = statewide_unalloced_indices.append(newyork_unalloced_indices)
    processed_df = raw_df.drop(unalloced_indices)

    # set index as counties
    processed_df = processed_df.set_index('state_county')

    # remove State, County Name, stateFIPS, countyFIPS columns
    if file_type == 'population':
        processed_df = processed_df.drop(columns=['countyFIPS', 'County Name', 'State'])
    else:
        processed_df = processed_df.drop(columns=['countyFIPS', 'County Name', 'State', 'stateFIPS'])

    # transpose data frame
    processed_df = processed_df.transpose()

    processed_df.to_csv(f'../../generated/us_{file_type}_counties.csv', index=True)
    return processed_df


def cumulativeCasesToVelocity(file_type, processed_df, save_csv=False):
    state_counties = processed_df.columns.tolist()
    velocity_df = pd.DataFrame().reindex_like(processed_df)

    for state_county in state_counties:
        prev_cases = 0
        for date in list(processed_df.index):
            velocity_df.loc[date,state_county] = processed_df.loc[date, state_county] - prev_cases
            prev_cases = processed_df.loc[date, state_county]

    if save_csv:
        velocity_df.to_csv(f'../../generated/us_velocity_{file_type}_counties.csv', index=True)
    return velocity_df


def findNegativeVelocityCounties(velocity_df, file_type):
    wrong_state_county_list = []
    for state_county in velocity_df.columns.tolist():
        is_data_wrong = (velocity_df[state_county].values < 0).any()
        if is_data_wrong:
            wrong_state_county_list.append(state_county)
    print(len(wrong_state_county_list))

    processed_velocity_df = velocity_df.drop(columns=wrong_state_county_list)
    processed_velocity_df.to_csv(f'../../generated/us_velocity_proc_{file_type}_counties.csv', index=True)
    return processed_velocity_df


if __name__ == "__main__":

    # processed_df = parseData(file_type='cases')

    for file_type in ['population', 'cases', 'deaths']:
        download_raw_data(file_type = file_type)
        processed_df = parseData(file_type=file_type)
        if file_type != 'population':
            velocity_df = cumulativeCasesToVelocity(file_type=file_type, processed_df = processed_df)
            velocity_df = pd.read_csv(f'../../generated/us_velocity_{file_type}_counties.csv', header=0, index_col=0)
            findNegativeVelocityCounties(velocity_df = velocity_df, file_type = file_type)


