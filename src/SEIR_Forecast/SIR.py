'''
Author: Junbong Jang
Date 11/27/2020

Creates SIR model that forecasts next day case/death numbers for one county
Take the parameters estimated from Bayesian Inference MCMC
'''

import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

def run(S0, I0, N, t, beta, gamma):
    # N is total number, t is the time points (in days)
    R0 = 0
    y0 = S0, I0, R0  # Initial conditions vector

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(sir_deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T

    return S, I, R


def sir_deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -1 * beta * S / N * I
    dIdt = beta * S / N * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def SIR_ground_truth(N, t, I, R, county):
    '''
    Creates graph of grount truth susceptible, infected and recovered in a county
    that resembles the SEIR model graph.
    :param N:
    :param t:
    :param I:
    :param R:
    :param county:
    :return:
    '''
    S = N-I
    plotsir(t, S, I, R, county)

    return I


def plotsir(t, S, I, R, county, save_path):
    plt.close()
    f, ax = plt.subplots(1,1,figsize=(10,4))
    ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    ax.plot(t, I, 'y', alpha=0.7, linewidth=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
    ax.plot(t, S + I, 'c--', alpha=0.7, linewidth=2, label='Total')

    ax.set_xlabel('Time (days)')
    ax.set_title(f'SIR model of {county}', fontsize='x-large')

    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
      ax.spines[spine].set_visible(False)

    plt.savefig(f'{save_path}sir_{county}.png')


def sir_forecast_a_county(S0, I0, N, t, beta, gamma, a_county, save_path):
    S, I, R = run(S0, I0, N, t, beta, gamma)
    # print('sir_forecast_a_county', S,I,R, N, I0, t, beta, gamma, a_county)
    # if a_county != '' and save_path != '':
        # plotsir(t, S, I, R, county, save_path)
    return I