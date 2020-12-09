import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from IPython.core.pylabtools import figsize
from timeseries_eval import *


def visualize_trace(samples, varname, N_SAMPLES):
    figsize(16, 10)

    plt.title(f'Distribution of {varname} not with {N_SAMPLES} samples')

    plt.hist(samples, histtype='stepfilled', 
             color = 'darkred', bins=30, alpha=0.8, density=True);
    plt.ylabel('Probability Density')
    
    plt.savefig(f'../../generated/plots/trace_{varname}.png')
    plt.clf()


def get_all_free_RVs_names(model):
    """
    Returns the names of all free parameters of the model
    Parameters
    ----------
        model: pm.Model instance
    Returns
    -------
        : list of variable names
    """
    varnames = [str(x).replace('_log__', '') for x in model.free_RVs]
    return varnames


def get_prior_distribution(model, x, varname):
    """
    Given a model and variable name, returns the prior distribution evaluated at x.
    Parameters
    ----------
    model: pm.Model instance
    x: list or array
    varname: string
    Returns
    -------
    : array
    """
    return np.exp(model[varname].distribution.logp(x).eval())


def plot_hist(model, trace, ax, varname, colors = ('tab:blue', 'tab:orange'), bins = 50):
    """
    Plots one histogram of the prior and posterior distribution of the variable varname.
    Parameters
    ----------
    model: pm.Model instance
    trace: trace of the model
    ax: matplotlib.axes  instance
    varname: string
    colors: list with 2 colornames
    bins:  number or array
        passed to np.hist
    Returns
    -------
    None
    """
    if len(trace[varname].shape) >= 2:
        print('Dimension of {} larger than one, skipping'.format(varname))
        ax.set_visible(False)
        return
    ax.hist(trace[varname], bins=bins, density=True, color=colors[1],
            label='Posterior')
    limits = ax.get_xlim()
    x = np.linspace(*limits, num=100)
    try:
        ax.plot(x, get_prior_distribution(model, x, varname), label='Prior',
                color=colors[0], linewidth=3)
    except:
        pass
    ax.set_xlim(*limits)
    ax.set_xlabel(varname)


def plot_cases(cluster_id, cluster_color, trace, current_cases_obs, future_cases_df, cluster_vel_case_forecast, cluster_all_vel_case_forecast,
               date_begin_sim, diff_data_sim, num_days_future, cluster_save_path, start_date_plot=None, end_date_plot=None,
               ylim=None, week_interval=None, colors = ('tab:blue', 'tab:green', 'tab:red', 'black')):
    """
    Plots the new cases, the fit, forecast and lambda_t evolution
    Parameters
    ----------
    trace : trace returned by model
    current_cases_obs : array
    date_begin_sim : datetime.datetime
    diff_data_sim : float
        Difference in days between the begin of the simulation and the data
    start_date_plot : datetime.datetime
    end_date_plot : datetime.datetime
    ylim : float
        the maximal y value to be plotted
    week_interval : int
        the interval in weeks of the y ticks
    colors : list with 2 colornames
    Returns
    -------
    figure, axes
    """

    def conv_time_to_mpl_dates(arr):
        return matplotlib.dates.date2num([datetime.timedelta(days=float(date)) + date_begin_sim for date in arr])

    new_cases_sim = trace['new_cases']
    len_sim = trace['lambda_t'].shape[1]

    if start_date_plot is None:
        start_date_plot = date_begin_sim + datetime.timedelta(days=diff_data_sim)
    if end_date_plot is None:
        end_date_plot = date_begin_sim + datetime.timedelta(days=len_sim)
    if ylim is None:
        # ylim = 1.6*np.max(current_cases_obs)
        ylim = 1.6*max(np.max(current_cases_obs), np.max(future_cases_df.sum(axis=1)))


    num_days_data = len(current_cases_obs)
    diff_to_0 = num_days_data + diff_data_sim
    date_data_end = date_begin_sim + datetime.timedelta(days=diff_data_sim + num_days_data)
    # num_days_future = (end_date_plot - date_data_end).days
    start_date_mpl, end_date_mpl = matplotlib.dates.date2num([start_date_plot, end_date_plot])

    if week_interval is None:
        week_inter_left = int(np.ceil(num_days_data/7/5))
        week_inter_right = int(np.ceil((end_date_mpl - start_date_mpl)/7/6))
    else:
        week_inter_left = week_interval
        week_inter_right = week_interval


    # fig, axes = plt.subplots(2, 1, figsize=(9, 7))  # , gridspec_kw={'height_ratios': [1, 3], 'width_ratios': [2, 3]}
    '''
    axes[0][0].set_visible(False)
    # --------------------------------------------------------------
    ax = axes[1][0]
    time_arr = np.arange(-len(current_cases_obs), 0)
    mpl_dates = conv_time_to_mpl_dates(time_arr) + diff_data_sim + num_days_data
    ax.plot(mpl_dates, current_cases_obs, 'd', markersize=6, label='Data', zorder=5, color=colors[0])
    cases_past = new_cases_sim[:, :num_days_data]
    percentiles = np.percentile(cases_past, q=2.5, axis=0), np.percentile(cases_past, q=97.5, axis=0)
    ax.plot(mpl_dates, np.median(cases_past, axis=0), color=colors[1], label='Fit (with 95% CI)')
    ax.fill_between(mpl_dates, percentiles[0], percentiles[1], alpha=0.3, color=colors[1])
    # ax.set_yscale('log')
    ax.set_ylabel('Number of new cases')
    ax.set_xlabel('Date')
    ax.legend()
    ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(interval = week_inter_left, byweekday=matplotlib.dates.SU))
    ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d'))
    ax.set_xlim(start_date_mpl)
    '''
    # --------------------------------------------------------------
    # effective growth rate change plot
    # ax = axes[0]
    #
    # time = np.arange(-diff_to_0 , -diff_to_0 + len_sim )
    # lambda_t = trace['lambda_t'][:, :]  # spreading rate
    # μ = trace['mu'][:, None]  # recovery rate
    # mpl_dates = conv_time_to_mpl_dates(time) + diff_data_sim + num_days_data
    #
    # ax.plot(mpl_dates, np.median(lambda_t - μ, axis=0), color=colors[1], linewidth=2)
    # ax.fill_between(mpl_dates, np.percentile(lambda_t - μ, q=2.5, axis=0), np.percentile(lambda_t - μ, q=97.5, axis=0),
    #                 alpha=0.15,
    #                 color=colors[1])
    #
    # ax.set_ylabel('effective\ngrowth rate $\lambda_t^*$')
    #
    # ylims = ax.get_ylim()
    # ax.hlines(0, start_date_mpl, end_date_mpl, linestyles=':')
    # delay = matplotlib.dates.date2num(date_data_end) - np.percentile(trace['delay'], q=75)
    #
    # ax.vlines(delay, ylims[0], ylims[1], linestyles='-', colors=colors[2])
    # ax.set_ylim(*ylims)
    # #ax.text(delay + 0.5, ylims[1] - 0.04*np.diff(ylims), 'unconstrained because\nof reporting delay', color='tab:red', verticalalignment='top')
    # #ax.text(delay - 0.5, ylims[1] - 0.04*np.diff(ylims), 'constrained\nby data', color='tab:red', horizontalalignment='right',
    # #        verticalalignment='top')
    # ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(interval = week_inter_right, byweekday=matplotlib.dates.SU))
    # ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
    # ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d'))
    # ax.set_xlim(start_date_mpl, end_date_mpl)
    
    # --------------------------------------------------------------
    # ax = axes[1]
    fig, ax = plt.subplots()
    
    # plot current data
    time1 = np.arange(-len(current_cases_obs), 0)
    mpl_dates = conv_time_to_mpl_dates(time1) + diff_data_sim + num_days_data
    ax.plot(mpl_dates, current_cases_obs, 'd', label='Reported Data', markersize=4, color='gray',
            zorder=5)
            
    # plot future data
    future_total_cases_series = future_cases_df.sum(axis=1)
    future_total_cases_time = np.arange(-len(future_total_cases_series) , 0)
    future_mpl_dates = conv_time_to_mpl_dates(future_total_cases_time) + num_days_data + len(future_total_cases_series)
    ax.plot(future_mpl_dates, future_total_cases_series, 'd', markersize=4, color='gray', zorder=5)

    # for a_county in future_cases_df:
    #     ax.plot(future_mpl_dates, future_cases_df[a_county], color=colors[0], linewidth=1, alpha=0.1)

    # plot fit line
    cases_past = new_cases_sim[:, :num_days_data]
    ax.plot(mpl_dates, np.median(cases_past, axis=0), '--', color=cluster_color, linewidth=1.5, label='Fit with 95% CI')
    percentiles = (
        np.percentile(cases_past, q=2.5, axis=0),
        np.percentile(cases_past, q=97.5, axis=0)
    )
    ax.fill_between(mpl_dates, percentiles[0], percentiles[1], alpha=0.2, color=cluster_color)

    # plot forecast line of a cluster
    time2 = np.arange(0, num_days_future)
    mpl_dates_fut = conv_time_to_mpl_dates(time2) + diff_data_sim + num_days_data
    cases_forecast = new_cases_sim[:, num_days_data:num_days_data+num_days_future].T
    
    median_cases_forecast = np.median(cases_forecast, axis=-1)
    percentiles = (
        np.percentile(cases_forecast, q=2.5, axis=-1),
        np.percentile(cases_forecast, q=97.5, axis=-1)
    )
    ax.plot(mpl_dates_fut, cluster_vel_case_forecast, color=cluster_color, linewidth=2, label='forecast', zorder=4)
    # ax.plot(mpl_dates_fut, median_cases_forecast, color=colors[1], linewidth=2, label='forecast with 75% and 95% CI', zorder=4)
    # ax.fill_between(mpl_dates_fut, percentles[0], percentiles[1], alpha=0.1, color=colors[1])
    # ax.fill_between(mpl_dates_fut, np.percentile(cases_forecast, q=12.5, axis=-1),
    #                 np.percentile(cases_forecast, q=87.5, axis=-1),
    #                 alpha=0.2, color=colors[1])

    # plot forecast line of cluster all
    text_x_position = 0.33
    if cluster_all_vel_case_forecast is not None:
        text_x_position = 0.45
        ax.plot(mpl_dates_fut, cluster_all_vel_case_forecast, color='black', linewidth=2, label='forecast from unclustered', zorder=3)
        # rmse_all_cluster = rmse_eval(future_total_cases_series.to_numpy().tolist(), cluster_all_vel_case_forecast)
        # ax.text(text_x_position, 0.80, 'RMSE:%.3f' % (rmse_all_cluster), color='black', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

    # rmse = rmse_eval(future_total_cases_series.to_numpy().tolist(), cluster_vel_case_forecast)
    # axes coordinates: (0, 0) is bottom left and (1, 1) is upper right # 0.39, 0.72
    # ax.text(text_x_position, 0.85, 'RMSE:%.3f' % (rmse), color=colors[1], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

    # ------------- Decoration ------------------
    ax.set_xlabel('Date')
    ax.set_ylabel(f"Daily infected\ncases in cluster {cluster_id}")
    if cluster_id == 1 or cluster_id == 'All':
        ax.legend(loc='upper left')
    ax.set_ylim(0, ylim)
    # func_format = lambda num, _: "${:.0f}\,$k".format(num / 1_000)
    # ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(func_format))
    ax.set_xlim(start_date_mpl, end_date_mpl)
    ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(interval = 2, byweekday=matplotlib.dates.MO))
    ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%d'))

    # plt.subplots_adjust(wspace=0.4, hspace=.3)
    plt.tight_layout()

    np.save(cluster_save_path + 'cases_forecast.npy', cases_forecast)
    np.save(cluster_save_path + 'cases_past.npy', cases_past)
    plt.savefig(cluster_save_path + 'plot_cases.png')
    plt.clf()

    return cases_forecast, cases_past