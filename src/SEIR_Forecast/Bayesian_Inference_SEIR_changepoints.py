'''
Author Junbong Jang
Date 8/14/2020

'''

import datetime

def get_change_points(final_date, final_change_date, cluster_id):
    print('get_change_points', final_date, cluster_id)

    if final_date == datetime.datetime(2020, 4, 30):
        prior_date_1 = datetime.datetime(2020, 3, 25)
        prior_date_2 = datetime.datetime(2020, 4, 1)
        prior_date_3 = datetime.datetime(2020, 4, 8)
            
    elif final_date == datetime.datetime(2020, 5, 15):
        prior_date_1 = datetime.datetime(2020, 4, 10)
        prior_date_2 = datetime.datetime(2020, 4, 17)
        prior_date_3 = datetime.datetime(2020, 4, 24)
    
    elif final_date == datetime.datetime(2020, 5, 31):
        prior_date_1 = datetime.datetime(2020, 4, 25)
        prior_date_2 = datetime.datetime(2020, 5, 1)
        prior_date_3 = datetime.datetime(2020, 5, 8)
        
    elif final_date == datetime.datetime(2020, 6, 15) or final_date == datetime.datetime(2020, 6, 18) or \
            final_date == datetime.datetime(2020, 6, 8) or final_date == datetime.datetime(2020, 6, 13):
        prior_date_1 = datetime.datetime(2020, 5, 10)
        prior_date_2 = datetime.datetime(2020, 5, 17)
        prior_date_3 = datetime.datetime(2020, 5, 24)
            
    elif final_date == datetime.datetime(2020, 6, 30):
        prior_date_1 = datetime.datetime(2020, 5, 25)
        prior_date_2 = datetime.datetime(2020, 6, 1)
        prior_date_3 = datetime.datetime(2020, 6, 8)
            
        
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
                 dict(pr_mean_date_begin_transient=final_change_date,
                      pr_sigma_date_begin_transient=1,
                      pr_median_lambda=0.2,
                      pr_sigma_lambda=0.5)]
                      
                      
    return change_points