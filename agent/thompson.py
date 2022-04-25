import numpy as np
from .base import Agent


class Thompson(Agent):

    def __init__(self, n_actions):
        super(Thompson, self).__init__("Thompson")
        self.n_actions = n_actions
        self.count_actions = None
        self.sum_reward = None
        self.chosen_action = None
        self.agent_type = 'Thompson'

        # self.n = 0  # the number of times this socket has been tried
        self.x = []  # list of all samples

        self.alpha = 1  # gamma shape parameter
        self.beta = 25  # gamma rate parameter

        self.mu_0 = 1  # the prior (estimated) mean
        self.v_0 = self.beta / (self.alpha + 1)  # the prior (estimated) variance


        self.init_flag = False
        self.init()

        # print(self.actions_dist_estimate)

    def init(self):
        # parameters estimation array -  row for each each action
        parameters_array = [np.zeros(self.n_actions, dtype=np.float64) for _ in range(5)]  # [mean, var, alpha, beta (gamma dist.), num_of_times_played]
        self.actions_dist_estimate = np.stack(parameters_array, axis=0)  # init means and vars arrays - mean and num of times are zeros
        self.actions_dist_estimate[1] += self.beta / (self.alpha + 1)  # init v to 25 for each action
        self.actions_dist_estimate[2] += self.alpha  # init alpha to 1 for each action
        self.actions_dist_estimate[3] += self.beta  # init beta to 50 for each action

        self.actions_dist_estimate = self.actions_dist_estimate.T
        self.samples = self.n_actions * [None] # [None], [None]...

        self.sum_reward = np.zeros(self.n_actions)
        if self.actions_dist_estimate is not None:
            self.init_flag = True

    def update_action_estimations(self, action, x):  # as depicted in the paper
        n = 1 #  num of observations
        v = self.actions_dist_estimate[action][4] # current action - num of times sampled

        self.actions_dist_estimate[action][2] += n / 2 # update alpha
        k = ((n * v / (v + n)) * (((x - self.actions_dist_estimate[action][0]) ** 2) / 2))
        self.actions_dist_estimate[action][3] += ((n * v / (v + n)) * (((x - self.actions_dist_estimate[action][0]) ** 2) / 2)) # update beta


        # estimate the variance - calculate the mean from the gamma hyper-parameters
        self.actions_dist_estimate[action][1] = self.actions_dist_estimate[action][3] / (self.actions_dist_estimate[action][2] + 1)  # the prior (estimated) variance
        if self.samples[action] is not None:
            self.samples[action].append(x)  # append the new value to the list of samples
        else:
            self.samples[action] = [x]

        self.actions_dist_estimate[action][4] += 1 # update num of times taken
        self.actions_dist_estimate[action][0] = np.array(self.samples[action]).mean() # update mean



    def _step(self, t):
        # print([np.random.normal(mean, var) for mean,var in self.actions_dist_estimate.T])
        sample_from_est_distribution = []
        o = self.actions_dist_estimate.copy()
        for mean, var, alpha, beta, n in o:
            try:
                precision = np.random.gamma(alpha, 1 / beta)
            except:
                precision = 0
                # print(mean, var, alpha, beta, n)
            if precision == 0 or n == 0: precision = 0.001
            estimated_variance = 1 / precision
            sample_from_est_distribution += [np.random.normal(mean, np.sqrt(estimated_variance))]
        action = np.argmax(sample_from_est_distribution)
        self.chosen_action = action
        return action

    def _get_reward(self, action, reward, t):
        self.sum_reward[action] += reward
        self.update_action_estimations(action, reward)

