def get_change_points(final_date):
    if final_date == '4/30':
        if cluster_id == 4:
            prior_date_mild_dist_begin = datetime.datetime(2020, 4, 5)
            prior_date_strong_dist_begin = datetime.datetime(2020, 4, 15)

            change_points = [dict(pr_mean_date_begin_transient=prior_date_mild_dist_begin,
                                  pr_sigma_date_begin_transient=1,
                                  pr_median_lambda=0.2,
                                  pr_sigma_lambda=0.5),
                             dict(pr_mean_date_begin_transient=prior_date_strong_dist_begin,
                                  pr_sigma_date_begin_transient=1,
                                  pr_median_lambda=0.2,
                                  pr_sigma_lambda=0.5)]
        elif cluster_id == 5:
            prior_date_mild_dist_begin = datetime.datetime(2020, 4, 5)
            prior_date_strong_dist_begin = datetime.datetime(2020, 4, 22)

            change_points = [dict(pr_mean_date_begin_transient=prior_date_mild_dist_begin,
                                  pr_sigma_date_begin_transient=1,
                                  pr_median_lambda=0.2,
                                  pr_sigma_lambda=0.5),
                             dict(pr_mean_date_begin_transient=prior_date_strong_dist_begin,
                                  pr_sigma_date_begin_transient=1,
                                  pr_median_lambda=0.2,
                                  pr_sigma_lambda=0.5)]

        elif cluster_id == 6:
            prior_date_mild_dist_begin = datetime.datetime(2020, 4, 9)
            prior_date_strong_dist_begin = datetime.datetime(2020, 4, 22)
            prior_date_contact_ban_begin = datetime.datetime(2020, 4, 27)

            change_points = [dict(pr_mean_date_begin_transient=prior_date_mild_dist_begin,
                                  pr_sigma_date_begin_transient=1,
                                  pr_median_lambda=0.2,
                                  pr_sigma_lambda=0.5),
                             dict(pr_mean_date_begin_transient=prior_date_strong_dist_begin,
                                  pr_sigma_date_begin_transient=1,
                                  pr_median_lambda=0.2,
                                  pr_sigma_lambda=0.5),
                             dict(pr_mean_date_begin_transient=prior_date_contact_ban_begin,
                                  pr_sigma_date_begin_transient=1,
                                  pr_median_lambda=0.2,
                                  pr_sigma_lambda=0.5)]

        elif cluster_id == 8:
            prior_date_mild_dist_begin = datetime.datetime(2020, 4, 4)
            prior_date_strong_dist_begin = datetime.datetime(2020, 4, 15)
            prior_date_contact_ban_begin = datetime.datetime(2020, 4, 25)

            change_points = [dict(pr_mean_date_begin_transient=prior_date_mild_dist_begin,
                                  pr_sigma_date_begin_transient=1,
                                  pr_median_lambda=0.2,
                                  pr_sigma_lambda=0.5),
                             dict(pr_mean_date_begin_transient=prior_date_strong_dist_begin,
                                  pr_sigma_date_begin_transient=1,
                                  pr_median_lambda=0.2,
                                  pr_sigma_lambda=0.5),
                             dict(pr_mean_date_begin_transient=prior_date_contact_ban_begin,
                                  pr_sigma_date_begin_transient=1,
                                  pr_median_lambda=0.2,
                                  pr_sigma_lambda=0.5)]

        elif cluster_id == 9:
            prior_date_mild_dist_begin = datetime.datetime(2020, 4, 3)
            prior_date_strong_dist_begin = datetime.datetime(2020, 4, 12)
            prior_date_contact_ban_begin = datetime.datetime(2020, 4, 22)

            change_points = [dict(pr_mean_date_begin_transient=prior_date_mild_dist_begin,
                                  pr_sigma_date_begin_transient=1,
                                  pr_median_lambda=0.2,
                                  pr_sigma_lambda=0.5),
                             dict(pr_mean_date_begin_transient=prior_date_strong_dist_begin,
                                  pr_sigma_date_begin_transient=1,
                                  pr_median_lambda=0.2,
                                  pr_sigma_lambda=0.5),
                             dict(pr_mean_date_begin_transient=prior_date_contact_ban_begin,
                                  pr_sigma_date_begin_transient=1,
                                  pr_median_lambda=0.2,
                                  pr_sigma_lambda=0.5)]

        elif cluster_id == 10:
            prior_date_mild_dist_begin = datetime.datetime(2020, 4, 5)
            prior_date_strong_dist_begin = datetime.datetime(2020, 4, 13)
            prior_date_contact_ban_begin = datetime.datetime(2020, 4, 26)

            change_points = [dict(pr_mean_date_begin_transient=prior_date_mild_dist_begin,
                                  pr_sigma_date_begin_transient=1,
                                  pr_median_lambda=0.2,
                                  pr_sigma_lambda=0.5),
                             dict(pr_mean_date_begin_transient=prior_date_strong_dist_begin,
                                  pr_sigma_date_begin_transient=1,
                                  pr_median_lambda=0.2,
                                  pr_sigma_lambda=0.5),
                             dict(pr_mean_date_begin_transient=prior_date_contact_ban_begin,
                                  pr_sigma_date_begin_transient=1,
                                  pr_median_lambda=0.2,
                                  pr_sigma_lambda=0.5)]
    elif final_date == '5/31':
        if cluster_id == -1:
            prior_date_1 = datetime.datetime(2020, 4, 3)
            prior_date_2 = datetime.datetime(2020, 4, 24)
            prior_date_3 = datetime.datetime(2020, 5, 17)
            prior_date_4 = datetime.datetime(2020, 5, 25)

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

        if cluster_id == 1:
            prior_date_1 = datetime.datetime(2020, 4, 10)
            prior_date_2 = datetime.datetime(2020, 4, 24)
            prior_date_3 = datetime.datetime(2020, 5, 15)
            prior_date_4 = datetime.datetime(2020, 5, 25)

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
        elif cluster_id == 2:
            prior_date_1 = datetime.datetime(2020, 4, 3)
            prior_date_2 = datetime.datetime(2020, 4, 19)
            prior_date_3 = datetime.datetime(2020, 5, 6)
            prior_date_4 = datetime.datetime(2020, 5, 24)

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
        elif cluster_id == 3:
            prior_date_1 = datetime.datetime(2020, 4, 3)
            prior_date_2 = datetime.datetime(2020, 4, 19)
            prior_date_3 = datetime.datetime(2020, 5, 7)
            prior_date_4 = datetime.datetime(2020, 5, 22)

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

        if cluster_id == 4:
            prior_date_1 = datetime.datetime(2020, 4, 3)
            prior_date_2 = datetime.datetime(2020, 4, 19)
            prior_date_3 = datetime.datetime(2020, 5, 15)
            prior_date_4 = datetime.datetime(2020, 5, 25)

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
        elif cluster_id == 5:
            prior_date_1 = datetime.datetime(2020, 3, 24)
            prior_date_2 = datetime.datetime(2020, 4, 9)
            prior_date_3 = datetime.datetime(2020, 4, 23)
            prior_date_4 = datetime.datetime(2020, 5, 15)

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

        elif cluster_id == 6:
            prior_date_1 = datetime.datetime(2020, 4, 7)
            prior_date_2 = datetime.datetime(2020, 4, 26)
            prior_date_3 = datetime.datetime(2020, 5, 12)
            prior_date_4 = datetime.datetime(2020, 5, 24)

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

        elif cluster_id == 7:
            prior_date_1 = datetime.datetime(2020, 3, 30)
            prior_date_2 = datetime.datetime(2020, 4, 16)
            prior_date_3 = datetime.datetime(2020, 5, 9)
            prior_date_4 = datetime.datetime(2020, 5, 22)

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

        elif cluster_id == 8:
            prior_date_1 = datetime.datetime(2020, 4, 2)
            prior_date_2 = datetime.datetime(2020, 4, 14)
            prior_date_3 = datetime.datetime(2020, 5, 8)
            prior_date_4 = datetime.datetime(2020, 5, 19)

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

        elif cluster_id == 9:
            prior_date_1 = datetime.datetime(2020, 3, 27)
            prior_date_2 = datetime.datetime(2020, 4, 14)
            prior_date_3 = datetime.datetime(2020, 4, 29)
            prior_date_4 = datetime.datetime(2020, 5, 19)

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

        elif cluster_id == 10:
            prior_date_1 = datetime.datetime(2020, 4, 10)
            prior_date_2 = datetime.datetime(2020, 4, 24)
            prior_date_3 = datetime.datetime(2020, 5, 15)
            prior_date_4 = datetime.datetime(2020, 5, 25)

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
        elif cluster_id == 11:
            prior_date_1 = datetime.datetime(2020, 3, 27)
            prior_date_2 = datetime.datetime(2020, 4, 14)
            prior_date_3 = datetime.datetime(2020, 5, 12)
            prior_date_4 = datetime.datetime(2020, 5, 22)

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

        elif cluster_id == 12:
            prior_date_1 = datetime.datetime(2020, 3, 27)
            prior_date_2 = datetime.datetime(2020, 4, 14)
            prior_date_3 = datetime.datetime(2020, 5, 12)
            prior_date_4 = datetime.datetime(2020, 5, 22)

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