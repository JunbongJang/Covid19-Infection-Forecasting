'''
Date 7/24/2020

Referenced from 
https://github.com/Priesemann-Group/covid19_inference_forecast/blob/master/
'''
import datetime
import platform

import theano
theano.config.floatX = 'float64'  # necessary to prevent an dtype error
import theano.tensor as tt
import numpy as np
import pymc3 as pm
import os
import matplotlib.pyplot as plt

from src.SEIR_Forecast.Bayesian_Inference_plotting import *
import src.SEIR_Forecast.Bayesian_Inference_SEIR_helper as model_helper
    
    
def run(sir_model, N_SAMPLES, cluster_save_path):
    print('sample start')
    with sir_model:
        trace = pm.sample(N_SAMPLES, model=sir_model, step=pm.Metropolis(), progressbar=True)
    pm.save_trace(trace, cluster_save_path + 'sir_model.trace', overwrite=True)
    print('sample end')
    # -------- prepare data for visualization ---------------
    varnames = get_all_free_RVs_names(sir_model)
    #for varname in varnames:
        #visualize_trace(trace[varname][:, None], varname, N_SAMPLES)
        
    lambda_t = np.median(trace['lambda_t'][:, :], axis=0)
    μ = np.median(trace['mu'][:, None], axis=0)

    # -------- visualize histogram ---------------
    num_cols = 5
    num_rows = int(np.ceil(len(varnames)/num_cols))
    x_size = num_cols * 2.5
    y_size = num_rows * 2.5

    fig, axes = plt.subplots(num_rows, num_cols, figsize = (x_size, y_size),squeeze=False)
    i_ax = 0
    for i_row, axes_row in enumerate(axes):
        for i_col, ax in enumerate(axes_row):
            if i_ax >= len(varnames):
                ax.set_visible(False)
                continue
            else:
                plot_hist(sir_model, trace, ax, varnames[i_ax], 
                                         colors=('tab:blue', 'tab:green'))
            if i_col == 0:
                ax.set_ylabel('Density')
            if i_col == 0 and i_row == 0:
                ax.legend()
            i_ax += 1
    fig.subplots_adjust(wspace=0.25, hspace=0.4)
    plt.savefig(cluster_save_path + 'plot_hist.png')
    plt.clf()

    np.save(cluster_save_path + 'varnames.npy', varnames)
    np.save(cluster_save_path + 'SIR_params.npy', [lambda_t, μ])
    

def SIR_with_change_points(
    S_begin_beta,
    I_begin_beta,
    new_cases_obs,
    change_points_list,
    date_begin_simulation,
    num_days_sim,
    diff_data_sim,
    N,
    priors_dict=None,
    weekends_modulated=False
):
    """
        Parameters
        ----------
        new_cases_obs : list or array
            Timeseries (day over day) of newly reported cases (not the total number)
        change_points_list : list of dicts
            List of dictionaries, each corresponding to one change point.
            Each dict can have the following key-value pairs. If a pair is not provided,
            the respective default is used.
                * pr_mean_date_begin_transient :     datetime.datetime, NO default
                * pr_median_lambda :                 number, same as default priors, below
                * pr_sigma_lambda :                  number, same as default priors, below
                * pr_sigma_date_begin_transient :    number, 3
                * pr_median_transient_len :          number, 3
                * pr_sigma_transient_len :           number, 0.3
        date_begin_simulation: datetime.datetime
            The begin of the simulation data
        num_days_sim : integer
            Number of days to forecast into the future
        diff_data_sim : integer
            Number of days that the simulation-begin predates the first data point in
            `new_cases_obs`. This is necessary so the model can fit the reporting delay.
            Set this parameter to a value larger than what you expect to find
            for the reporting delay.
            should be significantly larger than the expected delay,
            in order to always fit the same number of data points.
        N : number
            The population size. For Germany, we used 83e6
        priors_dict : dict
            Dictionary of the prior assumptions
            Possible key-value pairs (and default values) are:
                * pr_beta_I_begin :        number, default = 100
                * pr_median_lambda_0 :     number, default = 0.4
                * pr_sigma_lambda_0 :      number, default = 0.5
                * pr_median_mu :           number, default = 1/8
                * pr_sigma_mu :            number, default = 0.2
                * pr_median_delay :        number, default = 8
                * pr_sigma_delay :         number, default = 0.2
                * pr_beta_sigma_obs :      number, default = 10
                * week_end_days :          tuple,  default = (6,7)
                * pr_mean_weekend_factor : number, default = 0.7
                * pr_sigma_weekend_factor :number, default = 0.17
        weekends_modulated : bool
            Whether to add the prior that cases are less reported on week ends. Multiplies the new cases numbers on weekends
            by a number between 0 and 1, given by a prior beta distribution. The beta distribution is parametrised
            by pr_mean_weekend_factor and pr_sigma_weekend_factor
        weekend_modulation_type : 'step' or 'abs_sine':
            whether the weekends are modulated by a step function, which only multiplies the days given by  week_end_days
            by the week_end_factor, or whether the whole week is modulated by an abs(sin(x)) function, with an offset
            with flat prior.
        Returns
        -------
        : pymc3.Model
            Returns an instance of pymc3 model with the change points
    """
    if priors_dict is None:
        priors_dict = dict()

    default_priors = dict(
        pr_beta_I_begin=10000.0,
        pr_median_lambda_0=0.2,
        pr_sigma_lambda_0=0.5,
        pr_median_mu=1 / 8,
        pr_sigma_mu=0.2,
        pr_median_delay= 1.0,
        pr_sigma_delay=0.2,
        pr_beta_sigma_obs=5.0,
        week_end_days = (6,7),
        pr_mean_weekend_factor=0.7,
        pr_sigma_weekend_factor=0.17
    )
    default_priors_change_points = dict(
        pr_median_lambda=default_priors["pr_median_lambda_0"],
        pr_sigma_lambda=default_priors["pr_sigma_lambda_0"],
        pr_sigma_date_begin_transient=3.0,
        pr_median_transient_len=3.0,
        pr_sigma_transient_len=0.3,
        pr_mean_date_begin_transient=None,
    )

    if not weekends_modulated:
        del default_priors['week_end_days']
        del default_priors['pr_mean_weekend_factor']
        del default_priors['pr_sigma_weekend_factor']

    for prior_name in priors_dict.keys():
        if prior_name not in default_priors:
            raise RuntimeError(f"Prior with name {prior_name} not known")
    for change_point in change_points_list:
        for prior_name in change_point.keys():
            if prior_name not in default_priors_change_points:
                raise RuntimeError(f"Prior with name {prior_name} not known")

    for prior_name, value in default_priors.items():
        if prior_name not in priors_dict:
            priors_dict[prior_name] = value
            # print(f"{prior_name} was set to default value {value}")
    for prior_name, value in default_priors_change_points.items():
        for i_cp, change_point in enumerate(change_points_list):
            if prior_name not in change_point:
                change_point[prior_name] = value
                # print(f"{prior_name} of change point {i_cp} was set to default value {value}")

    if num_days_sim < len(new_cases_obs) + diff_data_sim:
        raise RuntimeError(
            "Simulation ends before the end of the data. Increase num_days_sim."
        )

    # ------------------------------------------------------------------------------ #
    # Model and prior implementation
    # ------------------------------------------------------------------------------ #

    with pm.Model() as model:
        # all pm functions now apply on the model instance
        # true cases at begin of loaded data but we do not know the real number
        I_begin = pm.Normal(name="I_begin", mu=I_begin_beta, sigma=I_begin_beta/10)
        S_begin = pm.Normal(name="S_begin", mu=S_begin_beta, sigma=S_begin_beta/10)
        # S_begin = N - I_begin

        # I_begin_print = tt.printing.Print('I_begin')(I_begin)
        # S_begin_print = tt.printing.Print('S_begin')(S_begin)
        # fraction of people that are newly infected each day
        lambda_list = []
        lambda_list.append(
            pm.Lognormal(
                name="lambda_0",
                mu=np.log(priors_dict["pr_median_lambda_0"]),
                sigma=priors_dict["pr_sigma_lambda_0"],
            )
        )
        for i, cp in enumerate(change_points_list):
            lambda_list.append(
                pm.Lognormal(
                    name=f"lambda_{i + 1}",
                    mu=np.log(cp["pr_median_lambda"]),
                    sigma=cp["pr_sigma_lambda"],
                )
            )

        # list of start dates of the transient periods of the change points
        tr_begin_list = []
        dt_before = date_begin_simulation
        for i, cp in enumerate(change_points_list):
            dt_begin_transient = cp["pr_mean_date_begin_transient"]
            if dt_before is not None and dt_before > dt_begin_transient:
                raise RuntimeError("Dates of change points are not temporally ordered")

            prior_mean = (dt_begin_transient - date_begin_simulation).days # - 1  # convert the provided date format (argument) into days (a number)

            tr_begin = pm.Normal(
                name=f"transient_begin_{i}",
                mu=prior_mean,
                sigma=cp["pr_sigma_date_begin_transient"],
            )
            tr_begin_list.append(tr_begin)
            dt_before = dt_begin_transient

        # same for transient times
        tr_len_list = []
        for i, cp in enumerate(change_points_list):
            tr_len = pm.Lognormal(
                name=f"transient_len_{i}",
                mu=np.log(cp["pr_median_transient_len"]),
                sigma=cp["pr_sigma_transient_len"],
            )
            tr_len_list.append(tr_len)

        # build the time-dependent spreading rate
        lambda_t_list = [lambda_list[0] * tt.ones(num_days_sim)]
        lambda_before = lambda_list[0]

        for tr_begin, tr_len, lambda_after in zip(
            tr_begin_list, tr_len_list, lambda_list[1:]
        ):
            lambda_t = model_helper.smooth_step_function(
                start_val=0,
                end_val=1,
                t_begin=tr_begin,
                t_end=tr_begin + tr_len,
                t_total=num_days_sim,
            ) * (lambda_after - lambda_before)
            lambda_before = lambda_after
            lambda_t_list.append(lambda_t)
        lambda_t = sum(lambda_t_list)

        # fraction of people that recover each day, recovery rate mu
        mu = pm.Lognormal(
            name="mu",
            mu=np.log(priors_dict["pr_median_mu"]),
            sigma=priors_dict["pr_sigma_mu"],
        )

        # delay in days between contracting the disease and being recorded
        # delay = pm.Lognormal(
        #     name="delay",
        #     mu=np.log(priors_dict["pr_median_delay"]),
        #     sigma=priors_dict["pr_sigma_delay"],
        # )

        # prior of the error of observed cases
        sigma_obs = pm.HalfCauchy("sigma_obs", beta=priors_dict["pr_beta_sigma_obs"])

        # -------------------------------------------------------------------------- #
        # training the model with loaded data provided as argument
        # -------------------------------------------------------------------------- #

        S, I, new_I = _SIR_model(
            lambda_t=lambda_t, mu=mu, S_begin=S_begin, I_begin=I_begin, N=N
        )

        # ignore this delay
        # new_cases_inferred = model_helper.delay_cases(
        #     new_I_t=new_I,
        #     len_new_I_t=num_days_sim,
        #     len_out=num_days_sim - diff_data_sim,
        #     delay=delay,
        #     delay_diff=diff_data_sim,
        # )
        new_cases_inferred = new_I

        # likelihood of the model:
        # observed cases are distributed following studentT around the model.
        # we want to approximate a Poisson distribution of new cases.
        # we choose nu=4 to get heavy tails and robustness to outliers.
        # https://www.jstor.org/stable/2290063
        num_days_data = new_cases_obs.shape[-1]
        pm.StudentT(
            name="_new_cases_studentT",
            nu=4,
            mu=new_cases_inferred[:num_days_data],
            sigma=tt.abs_(new_cases_inferred[:num_days_data] + 1) ** 0.5
            * sigma_obs,  # +1 and tt.abs to avoid nans
            observed=new_cases_obs,
        )

        # add these observables to the model so we can extract a time series of them
        # later via e.g. `model.trace['lambda_t']`
        pm.Deterministic("lambda_t", lambda_t)
        pm.Deterministic("new_cases", new_cases_inferred)
    return model



def _SIR_model(lambda_t, mu, S_begin, I_begin, N):
    """
        Implements the susceptible-infected-recovered model
        Parameters
        ----------
        lambda_t : ~numpy.ndarray
            time series of spreading rate, the length of the array sets the
            number of steps to run the model for
        mu : number
            recovery rate
        S_begin : number
            initial number of susceptible at first time step
        I_begin : number
            initial number of infected
        N : number
            population size
        Returns
        -------
        S : array
            time series of the susceptible
        I : array
            time series of the infected
        new_I : array
            time series of the new infected
    """
    new_I_0 = tt.zeros_like(I_begin)

    def next_day(lambda_t, S_t, I_t, _, mu, N):
        new_I_t = lambda_t / N * I_t * S_t
        S_t = S_t - new_I_t
        I_t = I_t + new_I_t - mu * I_t
        I_t = tt.clip(I_t, 0, N)  # for stability
        return S_t, I_t, new_I_t

    # theano scan returns two tuples, first one containing a time series of
    # what we give in outputs_info : S, I, new_I
    outputs, _ = theano.scan(
        fn=next_day,
        sequences=[lambda_t],
        outputs_info=[S_begin, I_begin, new_I_0],
        non_sequences=[mu, N],
    )
    return outputs
    
    