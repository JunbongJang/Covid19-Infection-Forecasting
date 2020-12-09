'''
Author Junbong Jang
Date 10/16/2020

To visualize the timeseries data in line graphs and
Plot histogram, bar or violin for the prediction evaluation metrics
'''

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def visualize_trend(ax, total_cases_series, cases_df, days, chosen_cluster, Rnot_index, Rsquared, Rnot):
                         
    # plot lines for counties
    for a_col in cases_df.columns:
        ax.plot(days, cases_df[a_col].values.tolist(), linewidth=1, alpha=0.2) # divide by constant for visualization
    
    # plot a line for mean values of all counties
    main_line = ax.plot(days, total_cases_series.values.tolist(), linewidth=3)
    
    # xtick decoration
    from matplotlib.ticker import (MultipleLocator, NullLocator)
    dates = cases_df.index.tolist()
    
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
    plt.ylabel('Daily infected cases', fontsize='large')

    return main_line


def find_Rnot_index(total_cases_series):
    Rnot_max_index = np.argmax(total_cases_series) + 1
    temp_max_cases = 0
    temp_max_counter = 0
    tolerance = 0.01
    for Rnot_index, one_cases in enumerate(total_cases_series):
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


def visualize_trend_with_r_not(chosen_cluster_id, cluster_cases_df, root_save_path):
    # min-max normalize data
    # for a_column in cluster_cases_df.columns:
    #     cluster_cases_df[a_column] = (cluster_cases_df[a_column] - cluster_cases_df[a_column].min()) / (cluster_cases_df[a_column].max() - cluster_cases_df[a_column].min())
    
    total_cluster_cases_series = cluster_cases_df.sum(axis=1)
    days = np.linspace(1, total_cluster_cases_series.size, total_cluster_cases_series.size)

    # find Rnot_index
    '''
    # process data by summation, log, shortening
    log_total_cases_series = np.log(total_cluster_cases_series.values)
    Rnot_index = find_Rnot_index(total_cluster_cases_series)
    print('Rnot_index', Rnot_index)
    
    # shortening using Rnot_index
    if np.any(np.isneginf(log_total_cases_series)):
        score_first = -1
        coef_first = -1
    else:
        infectious_period = 6.6
        shortened_mean_summed_cases_series = log_total_cases_series[:Rnot_index]
        shortened_days = days[:Rnot_index]
        score, coef, intercept = fit_R(shortened_days.reshape(-1, 1), shortened_mean_summed_cases_series.reshape(-1, 1))
        score_first = round(score, 3)
        coef_first = round(coef * infectious_period + 1, 3)
    '''
    score_first = 0
    coef_first = 0

    fig, ax = plt.subplots()
    main_line = visualize_trend(ax, total_cluster_cases_series, cluster_cases_df, days, chosen_cluster_id,
                                      Rnot_index=0,
                                      Rsquared=score_first, Rnot=coef_first)

    # draw curve fit line
    '''
    if not np.any(np.isneginf(log_total_cases_series)):
        shortened_mean_summed_cases_series = log_total_cases_series[:Rnot_index]
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


# ------------------ Visualize Evaluation metrics -----------------------------------
def bar_eval_clusters_compare(clustered_rmse_list, unclustered_rmse_list,
                              clustered_average_rmse, unclustered_average_rmse, metric_type, root_save_path):
    # process data
    assert len(clustered_rmse_list) == len(unclustered_rmse_list)
    total_rmse_list = clustered_rmse_list + unclustered_rmse_list

    cluster_id_list = [i for i in range(len(clustered_rmse_list))] + [i for i in range(len(unclustered_rmse_list))]
    clustered_type_list = ['clustered' for i in range(len(clustered_rmse_list))] + ['unclustered' for i in range(len(unclustered_rmse_list))]
    rmse_df = pd.DataFrame(list(zip(total_rmse_list, cluster_id_list, clustered_type_list)),
                 columns=['rmse', 'cluster_id', 'cluster_type'])

    # visualize data
    plot_colors = sns.color_palette('muted')
    sns.barplot( data=rmse_df, x="cluster_id", y="rmse", hue="cluster_type", palette=plot_colors )
    plt.xlabel("Cluster ID")
    plt.ylabel(metric_type)
    plt.legend(loc='upper left')
    plt.title(f"Comparison of {metric_type} per cluster")

    ax = plt.gca()
    ax.text(0.97, 0.93, f'Average {metric_type}:%.3f' % clustered_average_rmse, color=plot_colors[0], horizontalalignment='right',
            verticalalignment='bottom', transform=ax.transAxes)
    ax.text(0.97, 0.88, f'Average {metric_type}:%.3f' % unclustered_average_rmse, color=plot_colors[3], horizontalalignment='right',
            verticalalignment='bottom', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(root_save_path + f'{metric_type}_all_clusters_compare_bar.png')
    plt.close()


def bar_eval_clusters(clusters_num, cluster_eval_mean_list, average_eval, metric_type, cluster_mode, cluster_type, root_save_path):
    ax = plt.gca()
    ax.grid(zorder=0, axis='y')
    ax.bar(range(clusters_num),cluster_eval_mean_list)
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel(f'{metric_type} per cluster')
    ax.set_title(f'{metric_type} {cluster_mode} counties with {cluster_type}')
    ax.axhline(average_eval, color='red')
    ax.text(0.97, 0.94, f'Average {metric_type}:%.3f' % average_eval, color='tab:red', horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
    # if metric_type == 'RMSE':
    #     ax.set_ylim(0, 120)

    # elif metric_type == 'R^2':
    #     ax.set_ylim(0, 0.25)
    #
    # elif metric_type == 'WAPE' or metric_type == 'MAPE':
    #     ax.set_ylim(0, 350)

    plt.tight_layout()
    plt.savefig(root_save_path + f'{metric_type}_{cluster_mode}_bar.png')
    plt.close()


def violin_eval_clusters(eval_per_cluster_list, metric_type, root_save_path):
    ax = sns.violinplot(data=eval_per_cluster_list, inner='box', cut=0)
    eval_length_per_cluster = []
    for eval_per_cluster in eval_per_cluster_list:  # count number of counties per cluster
        eval_length_per_cluster.append(len(eval_per_cluster))

    for i, v in enumerate(eval_length_per_cluster):
        ax.text(i/len(eval_length_per_cluster)+0.05, 0.95, str(v), horizontalalignment='center', color='black', fontweight='bold', transform=ax.transAxes)

    plt.title(f'{metric_type} distribution per cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel(metric_type)
    # if metric_type == 'MSE':
    #     ax.set_ylim(0, 0.55)
    # elif metric_type == 'R^2':
    #     ax.set_ylim(0, 1)
    # elif metric_type == 'WAPE' or metric_type == 'MAPE':
    #     ax.set_ylim(0, 3000)

    plt.tight_layout()
    plt.savefig(root_save_path + f'{metric_type}_violin.png')
    plt.close()
    
    
def histogram_clusters(clusters, clusters_num, root_save_path):
    ax = plt.gca()
    sns.displot(clusters, bins=clusters_num)
    plt.title('Counties in each cluster')
    plt.xlabel('Cluster ID')
    # ax.set_ylim(top = 900)

    plt.tight_layout()
    plt.savefig(root_save_path + f'cluster_histogram.png')
    plt.close()