from env.base import Environment
from agent.eps_greedy import EpsilonGreedy
from agent.exp_exp import ExploreExploit, ExploreExploitOptimal
from agent.succ_elim import SuccessiveElimination
from agent.ucb1 import UCB1
from agent.ucb2 import UCB2
from agent.kalman import Kalman
from agent.thompson import Thompson

from sim.experiment import Experiment
from sim.utils import plot, plot_regrets, plot_arms
import numpy as np
from datetime import datetime
import pickle




def get_envs(bandit, agents, args):
    envs = [Environment(bandit, agent, delay=args.delay) for agent in agents]
    return envs


def run_epsilon_on_iid(bandit, epsilons, args):
    agents = [EpsilonGreedy(bandit.n_actions, epsilon) for epsilon in epsilons]
    envs = get_envs(bandit, agents, args)
    experiment = Experiment(envs, bandit, "eps-greedy", args)
    plot(experiment)


def run_exp_exp_on_iid(bandit, n_explores, args):
    agents = [ExploreExploit(bandit.n_actions, n_explore) for n_explore in n_explores]
    envs = get_envs(bandit, agents, args)
    experiment = Experiment(envs, bandit, "exp-exp", args)
    plot(experiment)


def run_exp_exp_opt_on_iid(bandit, args):
    return run_regret_experiment(bandit, ExploreExploitOptimal, args, "exp-exp-opt")


def run_regret_experiment(bandit, Agent, args, name):
    n_runs = args.n_runs
    n_steps = args.n_steps
    Ts = np.linspace(0, n_steps, args.n_regret_eval + 1).astype(int)
    final_regrets = np.zeros(len(Ts))
    agent = None
    env = None
    for i, T in enumerate(Ts):
        if i == 0:
            continue
        agent = Agent(bandit.n_actions, T)
        env = Environment(bandit, agent, delay=args.delay)
        regrets_sum = 0
        for _ in range(args.n_runs):
            _, _, _, _, regrets = env.run(n_steps=T)
            regrets_sum += regrets[-1]
        final_regrets[i] = regrets_sum / n_runs

    now = datetime.now()
    current_time = now.strftime("%y%m%d_%H%M%S")
    filename = "{}_{}_{}_{}_steps_{}_runs".format(current_time, args.bandit, name + "-regrets", n_steps, n_runs)
    filepath = "data/" + filename + ".p"

    with open(filepath, 'wb') as handle:
        pickle.dump((Ts, final_regrets, agent.name), handle)

    experiment = Experiment([env], bandit, name, args)

    plot_regrets([Ts], [final_regrets], [agent.name])
    plot(experiment)

    return Ts, final_regrets, agent.name


def run_succ_elim_on_iid(bandit, args):
    return run_regret_experiment(bandit, SuccessiveElimination, args, "succ-elim")


def run_ucb1_fixed_steps_on_iid(bandit, args):
    agents = [UCB1(bandit.n_actions, n_steps=args.n_steps)]
    envs = get_envs(bandit, agents, args)
    experiment = Experiment(envs, bandit, "ucb1-fixed", args)
    plot(experiment)


def run_ucb1_on_iid(bandit, args):
    agents = [UCB1(bandit.n_actions)]
    envs = get_envs(bandit, agents, args)
    experiment = Experiment(envs, bandit, "ucb1", args)
    plot(experiment)


def run_ucb2_on_iid(bandit, alphas, args):
    agents = [UCB2(bandit.n_actions, alpha) for alpha in alphas]
    envs = get_envs(bandit, agents, args)
    experiment = Experiment(envs, bandit, "ucb2", args)
    plot(experiment)

def run_kalman_on_iid(bandit, args):
    agents = [Kalman(bandit.n_actions, args.kalman_obs_noise)]
    envs = get_envs(bandit, agents, args)
    experiment = Experiment(envs, bandit, "kalman", args)
    plot(experiment)

def run_thompson_on_iid(bandit, args):
    agents = [Thompson(bandit.n_actions)]
    envs = get_envs(bandit, agents, args)
    experiment = Experiment(envs, bandit, "thompson", args)
    plot(experiment)
    plot_arms(agents[0])


def run_all_on_iid(bandit, args):
    agents = [
        Kalman(bandit.n_actions, args.kalman_obs_noise),
        Thompson(bandit.n_actions),
        ExploreExploitOptimal(bandit.n_actions, args.n_steps),
        EpsilonGreedy(bandit.n_actions),
        SuccessiveElimination(bandit.n_actions, args.n_steps),
        UCB1(bandit.n_actions),
        UCB2(bandit.n_actions, 0.01),
        Kalman(bandit.n_actions, args.kalman_obs_noise),
        Thompson(bandit.n_actions)
    ]
    envs = get_envs(bandit, agents, args)
    experiment = Experiment(envs, bandit, "all", args)
    save_dir_path = experiment.save_dir_path if hasattr(experiment, "save_dir_path") else None
    plot(experiment, save_dir_path)

def run_our_exp(bandit, args, true_parameters = None): # Our Experiment with the 4 Agents
    agents = [
        Kalman(bandit.n_actions, args.kalman_obs_noise),
        Thompson(bandit.n_actions),
        UCB1(bandit.n_actions),
        SuccessiveElimination(bandit.n_actions, args.n_steps)
    ]
    envs = get_envs(bandit, agents, args)
    experiment = Experiment(envs, bandit, f"all_{args.kalman_unknown_str}", args)
    save_dir_path = experiment.save_dir_path if hasattr(experiment, "save_dir_path") else None
    plot(experiment, save_dir_path)
    for agent in agents:
        plot_arms(agent,true_parameters,save_dir_path, args)
