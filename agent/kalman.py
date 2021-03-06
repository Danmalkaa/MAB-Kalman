import numpy as np
from .base import Agent


class Kalman(Agent):

    def __init__(self, n_actions, est_observation_noise):
        super(Kalman, self).__init__("Kalman")
        self.n_actions = n_actions
        self.agent_type = 'Kalman'
        self.count_actions = None
        self.sum_reward = None
        self.chosen_action = None
        self.default_observation_noise = 3.0
        self.observation_noise = est_observation_noise if est_observation_noise > 0 else self.default_observation_noise
        self.init_flag = False
        self.init()


    def init(self):
        mean_vars_arrays = [np.zeros(self.n_actions, dtype=np.float64) for _ in range(2)]
        self.actions_dist_estimate = np.stack(mean_vars_arrays,axis=0) # init means and vars arrays
        self.actions_dist_estimate[1] += 25 # init var to 25 for each action
        self.sum_reward = np.zeros(self.n_actions)
        if self.actions_dist_estimate is not None:
          self.init_flag = True

    def update_action_estimations(self,action,reward): # as depicted in the paper
        mean,var = self.actions_dist_estimate.T[action]
        new_mean = (var*reward + self.observation_noise*mean)/(var + self.observation_noise)
        new_var = (var*self.observation_noise)/(var+self.observation_noise)
        self.actions_dist_estimate.T[action] = new_mean, new_var
        

    def _step(self, t):
        sample_from_est_distribution = [np.random.normal(mean, np.sqrt(var)) for mean,var in self.actions_dist_estimate.T]
        action = np.argmax(sample_from_est_distribution)
        self.chosen_action = action
        return action

    def _get_reward(self, action, reward, t):
        self.sum_reward[action] += reward
        self.update_action_estimations(action,reward)

