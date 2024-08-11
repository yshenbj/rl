import torch
from ale_wrapper import ALEWrapper
from policy_gradient import PolicyGradient
from deep_nn_policy import DeepNeuralNetworkPolicy
from stochastic_q_policy import StochasticQPolicy
from deep_q_function import DeepQFunction
from multi_armed_bandit.epsilon_decreasing import EpsilonDecreasing
from tests.plot import Plot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

#version = "Freeway-ramDeterministic-v4"
#policy_name = "Freeway.policy"
# version = "ALE/Frogger-ram-v5"
version = 'CartPole-v1'

mdp = ALEWrapper(version)

# Get number of actions and state size from gym action space
action_space = len(mdp.get_actions())
state_space = len(mdp.get_initial_state())

runs = 1
episodes = 500
all_rewards = []
for _ in range(runs):
    policy = DeepNeuralNetworkPolicy(state_space, action_space)
    learner = PolicyGradient(mdp, policy)
    rewards = learner.execute(episodes, max_episode_length=500)
    all_rewards.append(rewards)

labels = ["Policy gradient" + str(i) for i in range(runs)]
Plot.plot_cumulative_rewards(labels, all_rewards, smoothing_factor=0.0)

mdp = ALEWrapper(version=version, render_mode="human")
exec_rewards = mdp.execute_policy(policy, episodes=1)
