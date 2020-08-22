'''
Author Junbong Jang
Date 8/14/2020

'''

import datetime

def get_change_points(final_date, cluster_id):
    print('get_change_points', final_date, cluster_id)

    if final_date == '4/30/2020':
        prior_date_1 = datetime.datetime(2020, 3, 25)
        prior_date_2 = datetime.datetime(2020, 4, 5)
        prior_date_3 = datetime.datetime(2020, 4, 14)
        prior_date_4 = datetime.datetime(2020, 4, 25)
            
    elif final_date == '5/15/2020':
        prior_date_1 = datetime.datetime(2020, 3, 30)
        prior_date_2 = datetime.datetime(2020, 4, 10)
        prior_date_3 = datetime.datetime(2020, 4, 25)
        prior_date_4 = datetime.datetime(2020, 5, 5)
    
    elif final_date == '5/31/2020':
        prior_date_1 = datetime.datetime(2020, 4, 5)
        prior_date_2 = datetime.datetime(2020, 4, 25)
        prior_date_3 = datetime.datetime(2020, 5, 10)
        prior_date_4 = datetime.datetime(2020, 5, 21)
        
    elif final_date == '6/15/2020':
        prior_date_1 = datetime.datetime(2020, 4, 5)
        prior_date_2 = datetime.datetime(2020, 4, 25)
        prior_date_3 = datetime.datetime(2020, 5, 15)
        prior_date_4 = datetime.datetime(2020, 6, 5)
            
    elif final_date == '6/30/2020':
        prior_date_1 = datetime.datetime(2020, 4, 5)
        prior_date_2 = datetime.datetime(2020, 4, 25)
        prior_date_3 = datetime.datetime(2020, 5, 30)
        prior_date_4 = datetime.datetime(2020, 6, 20)
         
    elif final_date == '7/15/2020':
        prior_date_1 = datetime.datetime(2020, 4, 5)
        prior_date_2 = datetime.datetime(2020, 5, 30)
        prior_date_3 = datetime.datetime(2020, 6, 20)
        prior_date_4 = datetime.datetime(2020, 7, 5)
            
    elif final_date == '7/31/2020':
        prior_date_1 = datetime.datetime(2020, 4, 5)
        prior_date_2 = datetime.datetime(2020, 5, 30)
        prior_date_3 = datetime.datetime(2020, 6, 20)
        prior_date_4 = datetime.datetime(2020, 7, 21)
            
        
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