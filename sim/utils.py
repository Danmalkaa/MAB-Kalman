import numpy as np
import matplotlib.pyplot as plt
import os, pickle


def load_data(filename):
    data_path = os.path.join(os.path.dirname(__file__)[:-3], filename[2:])

    with open(data_path, 'rb') as handle:
        experiment = pickle.load(handle)

    return experiment


def plot_from_data(filename):
    if "regret" in filename:
        Ts, regrets, labels = load_data(filename)
        plot_regrets([Ts], [regrets], [labels])
        plt.show()
    else:
        experiment = load_data(filename)
        plot(experiment)


def plot_regrets(Ts_arr, final_regrets, labels, save_dir_path=None):
    fig = plt.figure()
    for Ts, final_regrets_item, label in zip(Ts_arr, final_regrets, labels):
        plt.loglog(Ts, final_regrets_item, label=label)
    plt.legend()
    plt.xlabel(r"Number of total time steps $T$")
    plt.ylabel(r"$Regret(T)$")
    if save_dir_path:
        fig.savefig(os.path.join(save_dir_path,'Regret(t).png'))
    plt.show(block=False)


def plot(experiment, save_dir_path=None):
    n_steps = experiment.n_steps
    n_actions = experiment.bandit.n_actions
    labels = experiment.labels

    actions, rewards, cum_rewards, cum_rewards_mean, regrets, final_regrets = experiment.get_results()

    if final_regrets is not None:
        Ts = np.linspace(0, n_steps, final_regrets.shape[1]).astype(int)
        Ts_arr = [Ts for _ in range(len(labels))]
        plot_regrets(Ts_arr, final_regrets, labels, save_dir_path)

    fig = plt.figure()
    for cum_rewards_mean_item, label in zip(cum_rewards_mean, labels):
        plt.plot(np.arange(n_steps), cum_rewards_mean_item, label=label)
    plt.legend()
    plt.xlabel(r"Number of time steps $t$")
    plt.ylabel(r"$\overline{Reward}(t)$")
    if save_dir_path:
        fig.savefig(os.path.join(save_dir_path,'Reward(t).png'))

    for actions_item, label in zip(actions, labels):
        fig = plt.figure()
        bottom_sum = np.zeros(n_steps)
        for action in range(n_actions):
            plt.title(label)
            count = actions_item[:, action]
            plt.fill_between(np.arange(n_steps), bottom_sum, count + bottom_sum, label=str(action))
            bottom_sum += count
        plt.legend()
        plt.xlabel("Number of time steps")
        plt.ylabel("Action chosen for each time step")
        if save_dir_path:
            fig.savefig(os.path.join(save_dir_path, f'{label}.png'))

    plt.show(block=False)


# Taken from
# https://github.com/WhatIThinkAbout/BabyRobot/blob/master/Multi_Armed_Bandits/Part%205b%20-%20Thompson%20Sampling%20using%20Conjugate%20Priors.ipynb

import numpy as np
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from IPython.core.pylabtools import figsize

norm = stats.norm
gamma = stats.gamma

# matplotlib setup
figsize(11.0, 10)
x = np.linspace(0.0, 16.0, 200)


# return the index of the largest value in the supplied list
# - arbitrarily select between the largest values in the case of a tie
# (the standard np.argmax just chooses the first value in the case of a tie)
def random_argmax(value_list):
    """ a random tie-breaking argmax"""
    values = np.asarray(value_list)
    return np.argmax(np.random.random(values.shape) * (values == values.max()))


def plot_pdf(trials, mean, variance, label, ymax=0, set_height=False):
    y = norm.pdf(x, mean, np.sqrt(variance))

    p = plt.plot(x, y, lw=2, label=label)
    c = p[0].get_markeredgecolor()
    plt.fill_between(x, y, 0, color=c, alpha=0.2)
    plt.legend()
    plt.autoscale(tight=True)

    plt.vlines(mean, 0, y[1:].max(), colors=c, linestyles="--", lw=2)
    if ymax == 0: ymax = y[1:].max() * 1.1

    if set_height:
        axes = plt.gca()
        axes.set_ylim([0, ymax])

    return ymax


def plot_gamma(trials, alpha, beta, label, ymax=0, precision=0):
    variance = beta / (alpha + 1)

    y = stats.gamma.pdf(x, a=alpha, scale=1 / beta)

    p = plt.plot(x, y, lw=2, label=f"var = {alpha / beta ** 2:.3f}")
    c = p[0].get_markeredgecolor()

    plt.fill_between(x, y, 0, color=c, alpha=0.2)

    # if supplied, show the true precision
    if precision > 0:
        plt.vlines(x=[precision], ymin=0, ymax=(y[1:].max() * 1.1), colors=c, linestyles="--", lw=2)

    plt.title(f"{trials} Trials - Mean Precision = {1 / variance:.2f}")
    plt.legend()
    plt.autoscale(tight=True)

    axes = plt.gca()
    axes.set_ylim([0, y[1:].max() * 1.1])


figsize(11.0, 10)

norm = stats.norm
x = np.linspace(0.0, 18.0, 200)


def plot_socket_pdfs(sockets):
    ymax = 0
    for index, socket in enumerate(sockets):
        # get the PDF of the socket using its true values
        y = norm.pdf(x, socket.μ, np.sqrt(socket.v))

        p = plt.plot(x, y, lw=2, label=f"{index}")
        c = p[0].get_markeredgecolor()
        plt.fill_between(x, y, 0, color=c, alpha=0.2)
        plt.legend()
        plt.autoscale(tight=True)

        plt.vlines(socket.μ, 0, y[1:].max(), colors=c, linestyles="--", lw=2)

        ymax = max(ymax, y[1:].max() * 1.05)

    axes = plt.gca()
    axes.set_ylim([0, ymax])

    plt.legend(title='Sockets')
    plt.title('Density Plot of Socket Outputs')
    plt.xlabel('Socket Output (seconds of charge)')
    plt.ylabel('Density')


def plot_socket(socket, ymax=0, title=None):
    ymax1 = plot_pdf(socket.n, socket.μ, socket.v, "True", ymax)
    ymax2 = plot_pdf(socket.n, socket.μ_0, socket.v_0, "Estimated", ymax)

    if title is None: title = f"{socket.n} Trials"
    plt.title(title)

    # if no vertical extent is set automatically add space at top
    # - chooses the max from the 2 plots and adds 5%
    if ymax == 0: ymax = max(ymax1, ymax2) * 1.05
    axes = plt.gca()
    axes.set_ylim([0, ymax])


def plot_arms(agent, true_vals=None, save_dir_path = None, args=None):
    if agent.agent_type == 'Thompson':
        mean, var, alpha, beta, n = agent.actions_dist_estimate.T
    elif agent.agent_type == 'Kalman':
        mean, var = agent.actions_dist_estimate
    else:
        return
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    # trials = sum(n)
    plt.title(f"{agent.agent_type}")
    x = np.linspace(np.min(mean)-5.0, np.max(mean)+5*np.sqrt(var[np.argmax(mean)]), 200)

    ymax = 0
    for index in range(agent.n_actions):
        # get the PDF of the arm using its estimates
        y = norm.pdf(x, mean[index], np.sqrt(var[index]))
        if true_vals is None:
            label = fr'$\mu  = {mean[index]:.2f},  \sigma^2 = {var[index]:.2f}$'
        else:
            label = fr'Arm {index+1} '+'\n' + \
                    fr' Estimated: $\mu  = {mean[index]:.2f},  \sigma^2 = {var[index]:.2f}$; ' '\n' + \
                    fr'True Vals: $\mu  = {true_vals[index][0]:.2f},  \sigma^2 = {true_vals[index][1]:.2f}$'

        p = plt.plot(x, y, lw=2, label=label)
        c = p[0].get_markeredgecolor()
        plt.fill_between(x, y, 0, color=c, alpha=0.2)
        plt.legend()
        plt.autoscale(tight=True)

        plt.vlines(mean[index], 0, y[1:].max(), colors=c, linestyles="--", lw=2)

        ymax = max(ymax, y[1:].max() * 1.05)
        # label = fr'$\mu  = {mean[index]},  \sigma^2 = {var[index]}$'
        # plt.text(mean[index] , y[1:].max() + 5, label, ha="center", va="bottom")

    axes = plt.gca()
    axes.set_ylim([0, ymax])
    if save_dir_path:
        fig.savefig(os.path.join(save_dir_path,f'Arm_distribution_{agent.agent_type}_Obs_var_{args.obs_noise}.png'))

    plt.show(block=False)