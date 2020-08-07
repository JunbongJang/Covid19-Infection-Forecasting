'''
Author: Junbong Jang
Date 4/6/2020

Parses NYTimes Covid-19 Data into Velocity data
'''
import pandas as pd
import time

def parseNYTimes(col_type):
    raw_data = pd.read_csv('../../assets/QCI-COVID-19/us-counties.csv', header=0, index_col=0)
    raw_data = raw_data.fillna(0)

    dates = list(set(raw_data.index))
    dates.sort()
    state_counties = list(set(raw_data[['state', 'county']].agg('_'.join, axis=1)))
    state_counties.sort()

    # row index: date
    # column: county, state
    processed_df = pd.DataFrame(columns=state_counties, index=dates)

    # for loop optimization
    raw_data['state_county'] = raw_data[['state', 'county']].agg('_'.join, axis=1)
    raw_data.sort_values(by=['state_county'])

    # for every date(row) and unique location(column),
    # see if the combination has any confirmed cases or deaths
    start_time = time.time()
    for date in dates:
        print(date)
        rows_of_date = raw_data.loc[[date]]
        rows_of_date_iter = rows_of_date.iterrows()
        cur_row = next(rows_of_date_iter)[1]
        for location in state_counties:
            if cur_row['state_county'] == location:
                print('found: ' + location)
                processed_df.loc[date, location] = cur_row[col_type]
                try:
                    cur_row = next(rows_of_date_iter)[1]
                except StopIteration:
                    break

        processed_df.to_csv('../../generated/QCI-COVID-19/' + col_type + '.csv', index=True)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    print('end')


def cumulativeCasesToVelocity(col_type):
    parsed_data = pd.read_csv('../../generated/QCI-COVID-19/{}.csv'.format(col_type), header=0, index_col=0)
    parsed_data = parsed_data.fillna(0)
    state_counties = parsed_data.columns.tolist()
    velocity_df = pd.DataFrame().reindex_like(parsed_data)

    for state_county in state_counties:
        prev_cases = 0
        for date in list(parsed_data.index):
            velocity_df.loc[date,state_county] = parsed_data.loc[date,state_county] - prev_cases
            prev_cases = parsed_data.loc[date,state_county]

    velocity_df.to_csv('../../generated/QCI-COVID-19/velocity_{}.csv'.format(col_type), index=True)


def varify_parsed_data(oldfile, newfile):
    with open(oldfile, 'r') as t1, open(newfile, 'r') as t2:
        fileone = t1.readlines()
        filetwo = t2.readlines()

    with open('../../generated/QCI-COVID-19/update.csv', 'w') as outFile:
        for line in filetwo:
            if line not in fileone:
                outFile.write(line)


def download_raw_data():
    import urllib.request
    url = "https://github.com/nytimes/covid-19-data/blob/master/us-counties.csv?raw=true"
    filename, headers = urllib.request.urlretrieve(url, filename="../../assets/QCI-COVID-19/us-counties.csv")
    print("download file location: ", filename)
    print("download headers: ", headers)


if __name__ == "__main__":
    # download_raw_data()
    # parseNYTimes(col_type='cases')
    # parseNYTimes(col_type='deaths')
    cumulativeCasesToVelocity(col_type='cases')  # necessary for running Hacks
    cumulativeCasesToVelocity(col_type='deaths')
    # varify_parsed_data('../../generated/QCI-COVID-19/cases.csv','../../generated/QCI-COVID-19/cases_old.csv')

