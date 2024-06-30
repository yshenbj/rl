import torch

from advantage_actor_critic import AdvantageActorCritic
from deep_nn_policy import DeepNeuralNetworkPolicy
from deep_value_function import DeepValueFunction
from ale_wrapper import ALEWrapper
from tests.plot import Plot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

#version = "Freeway-ramDeterministic-v4"
#policy_name = "Freeway.policy"
version = 'CartPole-v1'
policy_name = "CartPole.policy"

mdp = ALEWrapper(version=version)

# Get number of actions and state size from gym action space
action_space = len(mdp.get_actions())
state_space = len(mdp.get_initial_state())

runs = 5
episodes = 500
all_rewards = []
for _ in range(runs):

    # Instantiate the critic
    critic = DeepValueFunction(state_space, hidden_dim=32)

    # Instantiate the actor
    actor = DeepNeuralNetworkPolicy(state_space, action_space, hidden_dim=32)

    advantage_actor_critic = AdvantageActorCritic(mdp, actor, critic)
    rewards = advantage_actor_critic.execute(episodes, max_episode_length=500)

    all_rewards.append(rewards)

labels = ["Advantage actor critic "  + str(i) for i in range(runs)]
Plot.plot_cumulative_rewards(labels, all_rewards, smoothing_factor=0.9)
