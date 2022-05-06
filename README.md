# MAB-Kalman
Kalman Agent for Mutli Arm Bandit

This project is developed as final project in TAU-Advanced Machine Learning and Algorithmic Game Theory (AML & AGT),
with the following task:
- Choose and implement a paper, take different assumptions and analyse theoretical and empiric results.

It implements the explore and exploit algorithm, <img src="https://render.githubusercontent.com/render/math?math=$\epsilon$">-greedy, successive elimination, UCB1 and UCB2, Thompson Sampling and a Kalman Filter based approach. 
Implementation follows the algorithms described in Introduction to Multi-Armed Bandits by A.Slivkins [1].

Innovation - 
- Implelments the stationary KF-MANB method offered in [2]
- Adding a time-independent Reward Delay.


Furthermore - adds "Noisy" Reward arms environment.

Authors - 
Dan Malka;
Mohammad Suliman;
Nathan Sala

[1] - [https://arxiv.org/pdf/1904.07272.pdf]

[2] - [https://www.researchgate.net/publication/221049713_Solving_Non-Stationary_Bandit_Problems_by_Random_Sampling_from_Sibling_Kalman_Filters]

# Run Experiments
<code>
  python main.py --exp "experiment number" --bandit "type of bandit"  
</code>
  
Experiment numbers are as follows: 
  0. Explore-exploit algorithm
  1. Optimal explore-exploit algorithm
  2. Epsilon-greedy algorithm
  3. Successive elimination algorithm
  4. UCB1 algorithm
  5. UCB2 algorithm
  6. Kalman
  7. Thompson Sampling
  8. Running our Experiment - Kalman, Thompson, UCB1, Succesive-elimination


Types of bandits are:
- *normal*: bandit arms have Gaussian distributed rewards
- *normal_noise*: bandit arms have Gaussian distributed rewards and rewards are obtained with noise (<img src="https://render.githubusercontent.com/render/math?math=$\sigma^2_{obs} = 3.0$"> as Default)
- *bernoulli* (default): bandit arms have bernoulli distributed rewards
- *bernoulli periodic*: success probability of the bernoulli distribution oscillates as a sinusoid.

**Plot Experiments**
Experiments data is stored as pickle files under the data directory.

<code>
python main.py --plot .\data\"experiment_path".p
</code>


**Default Parameters - in cli.py, main.py**
10 Arms - i.e. |A| = 10
Reward Distributions - 

*Normal* - <img src="https://render.githubusercontent.com/render/math?math=$\mu_i = [0.5, 1.0..+0.5k] k\in [0,|A|] , \quad \sigma^2_i = [0.5, 1.0..+0.5k] k\in [0,|A|]$">

Observation Noise Default Variance = [(i/5.0)*0.50 for i in range(1,11)]

Kalman Default Obs. Variance = 3.0

*Bernoulli* - <img src="https://render.githubusercontent.com/render/math?math=$P_i = [0.4, 0.45, 0.5, 0.55, 0.6] $">

<img src="https://render.githubusercontent.com/render/math?math=$\sigma^2_{obs} = 3.0$"> 

<code>
  --exp 7 --bandit normal_noise --n_runs 100 --n_steps 3000 --regrets True --n_regret_eval 10 --delay 0
  
  --obs_noise -1.0 --kalman_obs_noise -1.0
</code>

 - obs_noise/kalman_obs_noise = -1 --> uses default.
 - 
*For more Information:*

<code>
  python main.py -h
</code>
  
