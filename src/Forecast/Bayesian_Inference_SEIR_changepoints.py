'''
Author Junbong Jang
Date 8/14/2020

'''

import datetime

def get_change_points(final_date, final_change_date, cluster_id):
    print('get_change_points', final_date, cluster_id)

    if final_date == '4/30/2020':
        prior_date_1 = datetime.datetime(2020, 3, 25)
        prior_date_2 = datetime.datetime(2020, 4, 1)
        prior_date_3 = datetime.datetime(2020, 4, 8)
            
    elif final_date == '5/15/2020':
        prior_date_1 = datetime.datetime(2020, 4, 10)
        prior_date_2 = datetime.datetime(2020, 4, 17)
        prior_date_3 = datetime.datetime(2020, 4, 24)
    
    elif final_date == '5/31/2020':
        prior_date_1 = datetime.datetime(2020, 4, 25)
        prior_date_2 = datetime.datetime(2020, 5, 1)
        prior_date_3 = datetime.datetime(2020, 5, 8)
        
    elif final_date == '6/15/2020':
        prior_date_1 = datetime.datetime(2020, 5, 10)
        prior_date_2 = datetime.datetime(2020, 5, 17)
        prior_date_3 = datetime.datetime(2020, 5, 24)
            
    elif final_date == '6/30/2020':
        prior_date_1 = datetime.datetime(2020, 5, 25)
        prior_date_2 = datetime.datetime(2020, 6, 1)
        prior_date_3 = datetime.datetime(2020, 6, 8)

    
    prior_date_4 = final_change_date
            
        
    change_points = [dict(pr_mean_date_begin_transient=prior_date_1,
                      pr_sigma_date_begin_transient=1,
                      pr_median_lambda=0.2,
                      pr_sigma_lambda=0.5),
                 dict(pr_mean_date_begin_transient=prior_date_2,
                      pr_sigma_date_begin_transient=1,
                      pr_median_lambda=0.2,
                      pr_sigma_lambda=0.5),
                 dict(pr_mean_date_begin_transient=prior_date_3,
                      pr_sigma_date_begin_transient=1,
                      pr_median_lambda=0.2,
                      pr_sigma_lambda=0.5),
                 dict(pr_mean_date_begin_transient=prior_date_4,
                      pr_sigma_date_begin_transient=1,
                      pr_median_lambda=0.2,
                      pr_sigma_lambda=0.5)]
                      
                      
    return change_points