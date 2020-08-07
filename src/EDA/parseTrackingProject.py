'''
Author: Junbong Jang
Date 4/30/2020

Parses Covid19 Tracking Project state-level data
'''

import urllib.request
import pandas as pd
import numpy as np
import time


def download_raw_data():
    filename = "../../assets/us_states.csv"
    url = 'https://covidtracking.com/api/v1/states/daily.csv'
    filename, headers = urllib.request.urlretrieve(url, filename=filename)
    print("download file location: ", filename)
    print("download headers: ", headers)


def parseData():
    print('parseData')
    raw_df = pd.read_csv('../../assets/us_states.csv', header=0, index_col=0)
    dates = np.unique(raw_df.index).tolist()
    dates.sort()
    states = np.unique(raw_df[['state']].values).tolist()
    states.sort()

    start_time = time.time()
    for file_type in ['positive','negative','pending','hospitalizedCumulative','hospitalizedCurrently',
                      'inIcuCurrently','inIcuCumulative','onVentilatorCurrently','onVentilatorCumulative',
                      'recovered','death']:
        # row index: date
        # column: state
        processed_df = pd.DataFrame(columns=states, index=dates)
        chosen_df = raw_df[['state',file_type]]

        # for every date(row) and unique location(column),
        # see if the combination has any confirmed cases or deaths
        for date in dates:
            print(date)
            rows_of_date = chosen_df.loc[[date]]
            rows_of_date_iter = rows_of_date.iterrows()
            cur_row = next(rows_of_date_iter)[1]
            for location in states:
                if cur_row['state'] == location:
                    # print('found: ' + location)
                    processed_df.loc[date, location] = cur_row[file_type]
                    try:
                        cur_row = next(rows_of_date_iter)[1]
                    except StopIteration:
                        break

        processed_df.to_csv(f'../../generated/us_{file_type}_states.csv', index=True)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    print('end')
    return processed_df


if __name__ == "__main__":
    download_raw_data()
    parseData()
