from sim.iid import *
from cli import parse_command
from sim.utils import plot_from_data, plot_arms
from bandit.independent import IndependentBandit
from bandit.arm.normal import NormalArm, NormalNoiseArm
from bandit.arm.bernoulli import BernoulliArm
from bandit.arm.bernoulli_periodic import BernoulliPeriodicArm
import multiprocessing
import time
import os

# Means and Vars
MEANS = [0.50 + 0.50*i for i in range(10)]
VARS = [0.5 for i in range(10)]

obs_noise_vars = [(i/5.0)*0.50 for i in range(1,11)]
parameters_tup = np.vstack((MEANS,VARS)).T


delay = [i for i in range(1,11)]

P_SUCCESSES = [0.4, 0.45, 0.5, 0.55, 0.6]
# parameters_tup = np.vstack((MEANS,VARS)).T


def get_normal_bandit(means, vars):
    arms = [NormalArm(mean, var) for mean, var in zip(means, vars)]
    return IndependentBandit(arms)

def get_normal_noise_bandit(means, vars, obs_var):
    arms = [NormalNoiseArm(mean, var, obs_var) for mean, var in zip(means, vars)]
    return IndependentBandit(arms)


def get_bernoulli_bandit(ps):
    arms = [BernoulliArm(p) for p in ps]
    return IndependentBandit(arms)


def get_periodic_bandit(p_min, p_max, period, n):
    arms = [BernoulliPeriodicArm(p_min, p_max, period, period * i / n) for i in range(n)]
    return IndependentBandit(arms)


def main(args):

    bandit = get_bernoulli_bandit(P_SUCCESSES)
    if args.bandit == "normal":
        bandit = get_normal_bandit(MEANS, VARS)
    elif args.bandit == "normal_noise":
        obs_noise = args.obs_noise if args.obs_noise > 0 else OBS_VAR
        bandit = get_normal_noise_bandit(MEANS, VARS, obs_noise)
    elif args.bandit == "periodic":
        bandit = get_periodic_bandit(0.3, 0.7, 100, 5)

    if args.plot != "":
        plot_from_data(args.plot)

    elif args.exp == 0:
        print("Explore-exploit algorithm")
        n_explores = [0, 5, 10, 50, 100, 150]
        run_exp_exp_on_iid(bandit, n_explores, args)

    elif args.exp == 1:
        print("Optimal explore-exploit algorithm")
        run_exp_exp_opt_on_iid(bandit, args)

    elif args.exp == 2:
        print("Epsilon-greedy algorithm")
        epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 0.0, None]
        run_epsilon_on_iid(bandit, epsilons, args)

    elif args.exp == 3:
        print("Successive elimination algorithm")
        run_succ_elim_on_iid(bandit, args)

    elif args.exp == 4:
        print("UCB1 algorithm")
        run_ucb1_on_iid(bandit, args)

    elif args.exp == 5:
        print("UCB2 algorithm")
        alphas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
        run_ucb2_on_iid(bandit, alphas, args)

    elif args.exp == 6:
        print("All algorithms")
        run_all_on_iid(bandit, args, parameters_tup)

    elif args.exp == 7:
        print("Kalman")
        run_kalman_on_iid(bandit, args)

    elif args.exp == 8:
        print("Thompson")
        run_thompson_on_iid(bandit, args)

def multi_proc_exp_1(obs): # Known Variance for Kalman
    args = parse_command()
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    args.obs_noise = obs
    args.kalman_obs_noise = obs
    main(args)
    print(f"Done Known Var Obs. Var = {obs}")

def multi_proc_exp_2(obs): # Estimated Variance for Kalman
    args = parse_command()
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    args.obs_noise = obs
    args.kalman_unknown_str = "kalman_estimate"
    main(args)
    print(f"Done Unknown Obs. Var = {obs}")

def multi_proc_exp_3(delay, obs=0.1): # Known Variance for Kalman With Delay
    args = parse_command()
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    args.obs_noise = obs
    args.kalman_obs_noise = obs
    args.delay = delay
    args.kalman_unknown_str = f"kalman_known_delay_{delay}" # add the name of exp to the file and folder
    main(args)
    print(f"Done Known Var Obs. Delay = {delay}")

if __name__ == '__main__':
    with multiprocessing.Pool(8) as p:
        # Known Obs Var for kalman
        p.map(multi_proc_exp_1, obs_noise_vars)

        # Unknown Obs Var for kalman
        p.map(multi_proc_exp_2, obs_noise_vars)

        # known Obs Var = 0.1 for kalman + changing Delay of Reward - delay ranges in [1,10] timesteps
        p.map(multi_proc_exp_3, delay)


