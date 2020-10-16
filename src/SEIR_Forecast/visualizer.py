'''
Author Junbong Jang
Date 10/16/2020

functions that visualize the results
'''

from matplotlib import pyplot as plt
import seaborn as sns

def visualize_trend(ax, mean_cases_series, vel_cases_df, days, chosen_cluster, Rnot_index,
                          Rsquared, Rnot):
                         
    # plot lines for counties
    for a_col in vel_cases_df.columns:
        ax.plot(days, (vel_cases_df[a_col]/20).values.tolist(), linewidth=1) # divide by 10 for visualization
    
    # plot a line for mean values of all counties
    main_line = ax.plot(days, mean_cases_series.values.tolist(), linewidth=3)
    
    # xtick decoration 
    from matplotlib.ticker import (MultipleLocator, NullLocator)
    dates = vel_cases_df.index.tolist()
    
    spaced_out_dates=['']  # empty string elem needed because first element is skipped when using set_major_locator
    for date_index in range(0, len(dates)):
        dates[date_index] = dates[date_index].strftime("%m/%d/%y")
        if date_index%10 == 0:
            spaced_out_dates.append(str(dates[date_index]).replace('/2020',''))
        if str(dates[date_index]) in ['04/30/20', '05/31/20', '06/30/20', '07/31/20']:
            plt.axvline(x=date_index)
    plt.xticks(range(len(spaced_out_dates)), spaced_out_dates, size='small', rotation = (45))

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(NullLocator())

    text_input = ''  # r'$R^2$ =' + str(Rsquared) + '\n' + r'$R_0$ = ' + str(Rnot)
    ax.text(0.12, 0.77, text_input, color='black',
            bbox=dict(facecolor='none', edgecolor='black', pad=10.0),
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax.transAxes)

    # plt.axvline(x=Rnot_index, color='k', linestyle='dotted')

    plt.title(f'Reported COVID-19 Case Trend in Cluster {chosen_cluster}', fontsize='x-large')
    plt.xlabel('Dates', fontsize='large')
    plt.ylabel('New Confirmed\nCases in each day', fontsize='large')

    return main_line


def find_Rnot_index(mean_cases_series):
    Rnot_max_index = np.argmax(mean_cases_series) + 1
    temp_max_cases = 0
    temp_max_counter = 0
    tolerance = 0.01
    for Rnot_index, one_cases in enumerate(mean_cases_series):
        temp_max_counter = temp_max_counter + 1
        if one_cases > temp_max_cases + tolerance:
            temp_max_cases = one_cases
            temp_max_counter = 0
        if temp_max_counter > 10:
            Rnot_index = Rnot_index - 10
            break
    if Rnot_index > Rnot_max_index:
        Rnot_index = Rnot_max_index
    return Rnot_index


def visualize_trend_with_r_not(chosen_cluster_id, cluster_vel_cases_df, root_save_path):
    # min-max normalize data
    for a_column in cluster_vel_cases_df.columns:
        cluster_vel_cases_df[a_column] = (cluster_vel_cases_df[a_column] - cluster_vel_cases_df[a_column].min()) / (cluster_vel_cases_df[a_column].max() - cluster_vel_cases_df[a_column].min())
    
    mean_cluster_vel_cases_series = cluster_vel_cases_df.mean(axis=1)

    # process data by summation, log, shortening
    days = np.linspace(1, mean_cluster_vel_cases_series.size, mean_cluster_vel_cases_series.size)
    log_mean_cases_series = np.log(mean_cluster_vel_cases_series.values)

    # find Rnot_index
    '''
    Rnot_index = find_Rnot_index(mean_cluster_vel_cases_series)
    print('Rnot_index', Rnot_index)
    shortening using Rnot_index
    if np.any(np.isneginf(log_mean_cases_series)):
        score_first = -1
        coef_first = -1
    else:
        infectious_period = 6.6
        shortened_mean_summed_cases_series = log_mean_cases_series[:Rnot_index]
        shortened_days = days[:Rnot_index]
        score, coef, intercept = fit_R(shortened_days.reshape(-1, 1), shortened_mean_summed_cases_series.reshape(-1, 1))
        score_first = round(score, 3)
        coef_first = round(coef * infectious_period + 1, 3)
    '''
    score_first = 0
    coef_first = 0

    fig, ax = plt.subplots()
    main_line = visualize_trend(ax, mean_cluster_vel_cases_series, cluster_vel_cases_df, days, chosen_cluster_id,
                                      Rnot_index=0,
                                      Rsquared=score_first, Rnot=coef_first)

    # draw curve fit line
    '''
    if not np.any(np.isneginf(log_mean_cases_series)):
        shortened_mean_summed_cases_series = log_mean_cases_series[:Rnot_index]
        shortened_days = days[:Rnot_index]
        curve_fit = np.polyfit(shortened_days, shortened_mean_summed_cases_series, 1)
        y_days = np.exp(curve_fit[0] * shortened_days) * np.exp(curve_fit[1])
        curve_fit_line = ax.plot(shortened_days, y_days, '--')

        plt.legend((main_line[0], curve_fit_line[0]), ('Average of all counties', 'Curve Fit'),
                loc='upper left')
    else:
        plt.legend(main_line, ['Average of all counties'], loc='upper left')
    '''
    fig.tight_layout()
    fig.savefig(root_save_path + 'reported_cases_trend.png', dpi=fig.dpi)
    plt.close()


def bar_rmse_clusters(clusters_num, cluster_rmse_mean_list, title_prefix, cluster_type, root_save_path, save_name, average_of_rmse):
    ax = plt.gca()
    ax.bar(range(clusters_num),cluster_rmse_mean_list)
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('RMSE per cluster')
    ax.set_title(title_prefix + ' ' + cluster_type)
    ax.axhline(average_of_rmse, color='red')
    ax.text(0.97, 0.94, 'Average RMSE:%.3f' % (average_of_rmse), color='tab:red', horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
    ax.set_ylim(0.075, 0.275)
    
    plt.savefig(root_save_path + f'{save_name}.png')
    plt.close()


def violin_mse_clusters(mse_per_cluster_list, root_save_path):
    ax = sns.violinplot(data=mse_per_cluster_list, cut=0)
    plt.title('MSE distribution per cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('MSE')
    plt.savefig(root_save_path + f'mse_violin.png')
    plt.close()
    
    
def histogram_clusters(clusters, root_save_path):
    sns.displot(clusters)
    plt.title('Counties in each cluster')
    plt.xlabel('Cluster ID')
    plt.tight_layout()
    plt.savefig(root_save_path + f'cluster_histogram.png')
    plt.close()