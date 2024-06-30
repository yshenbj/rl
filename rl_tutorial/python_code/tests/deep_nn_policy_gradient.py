from gridworld import GridWorld
from policy_gradient import PolicyGradient
from deep_nn_policy import DeepNeuralNetworkPolicy
from tests.plot import Plot


gridworld = GridWorld()
state_space = len(gridworld.get_initial_state())
action_space = len(gridworld.get_actions())
policy = DeepNeuralNetworkPolicy(state_space, action_space)
rewards = PolicyGradient(gridworld, policy).execute(episodes=2000)
gridworld_image = gridworld.visualise_stochastic_policy(policy)
Plot.plot_cumulative_rewards(["REINFORCE"], [rewards], smoothing_factor=0.8)

gridworld.visualise_policy(policy)
