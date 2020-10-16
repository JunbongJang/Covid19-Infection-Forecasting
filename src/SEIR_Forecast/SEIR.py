'''
Author: Junbong Jang
Date 4/30/2020

Creates SEIR model that forecasts next day case/death numbers for one county
Refered to https://towardsdatascience.com/infectious-disease-modelling-beyond-the-basic-sir-model-216369c584c4
'''

import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


def fit_R(days, infection_series):
    reg = LinearRegression().fit(days, infection_series)
    score = reg.score(days, infection_series) # coefficient of determination R^2
    coef = reg.coef_[0,0]
    intercept = reg.intercept_[0]

    return score, coef, intercept


def SEIR(N, t, lockdown_day, chosen_county):
    # N is total number, t is the time points (in days)
    D = 4.0  # infections lasts four days
    gamma = 1.0 / D # recovery rate
    delta = 1.0 / 5.0  # incubation period of five days

    R_0_start = 2.2
    R_0_end = 0.9
    def logistic_R_0(t):
        k = 0.2 # how quickly R_0 declines
        return (R_0_start - R_0_end) / (1 + np.exp(-k * (-t + lockdown_day))) + R_0_end

    def R_0(t):   #  the total number of people an infected person infects
        return R_0_start if t < lockdown_day else R_0_end

    def beta(t):  # infected person infects # other person per day
        return R_0_start * gamma

    S0, E0, I0, R0 = N - 1, 1, 0, 0  # initial conditions: one exposed
    y0 = S0, E0, I0, R0  # Initial conditions vector

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, delta))
    S, E, I, R = ret.T

    plotseird(t, S, E, I, R,chosen_county)

    return I


def SEIR_actual_graph(N, t, I, R, chosen_county):
    '''
    Creates graph of grount truth susceptible, infected and recovered in a county
    that resembles the SEIR model graph.
    :param N:
    :param t:
    :param I:
    :param R:
    :param chosen_county:
    :return:
    '''
    S = N-I
    plotsir(t, S, I, R, chosen_county)

    return I


def deriv(y, t, N, beta, gamma, delta):
    S, E, I, R = y
    dSdt = -1 * beta(t) * S * I / N
    dEdt = beta(t) * S * I / N - delta * E
    dIdt = delta * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


def plotsir(t, S, I, R,chosen_county):
    f, ax = plt.subplots(1,1,figsize=(10,4))
    ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    ax.plot(t, I, 'y', alpha=0.7, linewidth=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
    ax.plot(t, S + I, 'c--', alpha=0.7, linewidth=2, label='Total')

    ax.set_xlabel('Time (days)')
    ax.set_title(f'SIR model of {chosen_county}', fontsize='x-large')

    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
      ax.spines[spine].set_visible(False)
    plt.show()


def plotseird(t, S, E, I, R, chosen_county, D=None, L=None, R0=None, Alpha=None):
    f, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    ax.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')
    ax.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
    if D is not None:
        ax.plot(t, D, 'k', alpha=0.7, linewidth=2, label='Dead')
        ax.plot(t, S + E + I + R + D, 'c--', alpha=0.7, linewidth=2, label='Total')
    else:
        ax.plot(t, S + E + I + R, 'c--', alpha=0.7, linewidth=2, label='Total')

    ax.set_xlabel('Time (days)')
    ax.set_title(f'SIR model of {chosen_county}', fontsize='x-large')

    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend(borderpad=2.0)
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    if L is not None:
        plt.title("Lockdown after {} days".format(L))
    plt.show()

    if R0 is not None:
        f = plt.figure(figsize=(12, 4))

    if R0 is not None:
        # sp1
        ax1 = f.add_subplot(121)
        ax1.plot(t, R0, 'b--', alpha=0.7, linewidth=2, label='R_0')

        ax1.set_xlabel('Time (days)')
        ax1.title.set_text('R_0 over time')
        # ax.set_ylabel('Number (1000s)')
        # ax.set_ylim(0,1.2)
        ax1.yaxis.set_tick_params(length=0)
        ax1.xaxis.set_tick_params(length=0)
        ax1.grid(b=True, which='major', c='w', lw=2, ls='-')
        legend = ax1.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)

    if Alpha is not None:
        # sp2
        ax2 = f.add_subplot(122)
        ax2.plot(t, Alpha, 'r--', alpha=0.7, linewidth=2, label='alpha')

        ax2.set_xlabel('Time (days)')
        ax2.title.set_text('fatality rate over time')
        # ax.set_ylabel('Number (1000s)')
        # ax.set_ylim(0,1.2)
        ax2.yaxis.set_tick_params(length=0)
        ax2.xaxis.set_tick_params(length=0)
        ax2.grid(b=True, which='major', c='w', lw=2, ls='-')
        legend = ax2.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)

        plt.show()
