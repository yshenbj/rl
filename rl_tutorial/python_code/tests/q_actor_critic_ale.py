import torch

from q_actor_critic import QActorCritic
from deep_nn_policy import DeepNeuralNetworkPolicy
from old_deep_q_function import DeepQFunction
from ale_wrapper import ALEWrapper
from tests.plot import Plot

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
episodes = 1000
all_rewards = []
for _ in range(runs):

    # Instantiate the critic
    critic = DeepQFunction(state_space, action_space)

    # Instantiate the actor
    actor = DeepNeuralNetworkPolicy(state_space, action_space)

    learner = QActorCritic(mdp, actor, critic)
    rewards = learner.execute(episodes)

    all_rewards.append(rewards)

print(all_rewards)
labels = ["Q actor critic" + str(i) for i in range(runs)]
Plot.plot_cumulative_rewards(labels, all_rewards, smoothing_factor=0.0)

mdp = ALEWrapper(version=version, render_mode="human")
exec_rewards = mdp.execute_policy(actor, episodes=1)
