'''
Author Junbong Jang
Date 7/24/2020

Find SEIR model parameters through bayeisan Inference and Monte Carlo Markov Chain
'''

# https://github.com/WillKoehrsen/ai-projects/blob/master/markov_chain_monte_carlo/markov_chain_monte_carlo.ipynb
# pymc3 for Bayesian Inference, pymc built on t
import pymc3 as pm
import theano
from theano import tensor as tt
import scipy
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize


def bayesian_inference_SEIR(day_array, cluster_vel_cases_array, N_SAMPLES):
    # https://discourse.pymc.io/t/how-to-sample-efficiently-from-time-series-data/4928
    N_SAMPLES = 1000
    s0, e0, i0 = 100., 50., 25.
    st0, et0, it0 = [theano.shared(x) for x in [s0, e0, i0]]

    C = np.array([3, 5, 8, 13, 21, 26, 10, 3], dtype=np.float64)
    D = np.array([1, 2, 3, 7, 9, 11, 5, 1], dtype=np.float64)

    def seir_one_step(st0, et0, it0, beta, gamma, delta):
        bt0 = st0 * beta
        ct0 = et0 * gamma
        dt0 = it0 * delta
        
        st1 = st0 - bt0
        et1 = et0 + bt0 - ct0
        it1 = it0 + ct0 - dt0
        return st1, et1, it1

    with pm.Model() as model:
        beta = pm.Beta('beta', 2, 10)
        gamma = pm.Beta('gamma', 2, 10)
        delta = pm.Beta('delta', 2, 10)

        (st, et, it), updates = theano.scan(
            fn=seir_one_step,
            outputs_info=[st0, et0, it0],
            non_sequences=[beta, gamma, delta],
            n_steps=len(C))

        ct = pm.Binomial('c_t', et, gamma, observed=C)
        dt = pm.Binomial('d_t', it, delta, observed=D)

        trace = pm.sample(N_SAMPLES)
        print(trace)
        visualize_trace(trace["beta"][:, None], trace["gamma"][:, None], 
                        trace["delta"][:, None], N_SAMPLES)

    with model:
        bt = pm.Binomial('b_t', st, beta, shape=len(C))
        ppc_trace = pm.sample_posterior_predictive(trace, var_names=['b_t'])



def get_time_variant_R(day_array, cluster_cases_array, cluster_mean_population, N_SAMPLES):
    print(day_array)
    print(cluster_cases_array)
    print(cluster_mean_population)
    r_model = pm.Model()
    
    cluster_cases_array = np.array([3, 5, 6, 7, 5, 8, 6, 4,7,5], dtype=np.float64)
    day_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)

    with r_model:
        b_not = pm.Normal('b_not', mu=2.5, sigma=2)
        b_t = pm.Normal('b_t', mu=2.5, sigma=2)
        adaptation_t = pm.Normal('adaptation_t', mu=10, sigma=10)
        transition_t = pm.Normal('transition_t', mu=10, sigma=10)

        bt = pm.Deterministic('bt', b_not - 1 / 2 * (1 + pm.math.tanh(day_array - adaptation_t) / transition_t) * (b_not - b_t) )
        
        observed = pm.Normal('obs', mu=bt, sigma=1, observed=cluster_cases_array)

        # Using Metropolis Hastings Sampling
        # Sample from the posterior using the sampling method
        trace = pm.sample(N_SAMPLES, step=pm.Metropolis())
        print(trace)
        visualize_trace(trace["b_not"][:, None], trace["b_t"][:, None], 
                        trace["adaptation_t"][:, None], trace["transition_t"][:, None], N_SAMPLES)
        print('--------- end ----------')
        
        