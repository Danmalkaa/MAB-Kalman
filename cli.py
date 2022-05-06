import argparse


def parse_command():
    parser = argparse.ArgumentParser()
    _add_experiment_parser(parser)
    args = parser.parse_args()

    return args


def _add_experiment_parser(parser):
    o_parser = parser.add_argument_group(title='Experiment types')
    o_parser.add_argument('--plot', default="",
                          help="experiment data to plot")
    o_parser.add_argument('--exp', type=int, default=8,
                          help="experiment to run")
    o_parser.add_argument('--n_runs', type=int, default=100,
                          help="number of runs")
    o_parser.add_argument('--n_steps', type=int, default=3000,
                          help="number of steps")
    o_parser.add_argument('--regrets', type=bool, default=True,
                          help="plot regret against the number of rounds")
    o_parser.add_argument('--n_regret_eval', type=int, default=10,
                          help="number of experiments to evaluate regrets")
    o_parser.add_argument('--bandit', type=str, default="normal_noise",
                          help="type of bandit")
    o_parser.add_argument('--delay', type=int, default=0,
                          help="delay of reward")
    o_parser.add_argument('--obs_noise', default="-1.0",
                          type=float, help="Reward Noise Variance")
    o_parser.add_argument('--kalman_obs_noise', default="-1.0",
                          type=float, help="Kalman known or estimated Reward Noise Variance")
    o_parser.add_argument('--kalman_unknown_str', type=str, default="kalman_known",
                          help="type of bandit")