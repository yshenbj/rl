import torch

from freeway_abstraction import FreewayAbstraction
from freeway import Freeway
from qlearning import QLearning
from deep_q_function import DeepQFunction
from stochastic_q_policy import StochasticQPolicy
from ale_wrapper import ALEWrapper
from multi_armed_bandit.epsilon_decreasing import EpsilonDecreasing
from tests.plot import Plot


# if GPU is to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

#version = "Freeway-ramDeterministic-v4"
#policy_name = "Freeway.policy"
version = 'CartPole-v1'

mdp = ALEWrapper(version=version)

# Get number of actions and state size from gym action space
action_space = len(mdp.get_actions())
state_space = len(mdp.get_initial_state())

runs = 5
episodes=1000
all_rewards = []
for _ in range(runs):
    qfunction = DeepQFunction(state_space, action_space)
    learner = QLearning(mdp, EpsilonDecreasing(), qfunction)

    rewards = learner.execute(episodes)
    all_rewards.append(rewards)

labels = ["Deep Q learner " + str(i) for i in range(runs)]
Plot.plot_cumulative_rewards(labels, all_rewards, smoothing_factor=0.0)

policy = StochasticQPolicy(qfunction, EpsilonDecreasing(epsilon=0.04, alpha=1., lower_bound=0.0))

mdp = ALEWrapper(version=version, render_mode="human")
exec_rewards = mdp.execute_policy(policy, episodes=1)
