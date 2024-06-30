from ale_wrapper import ALEWrapper
from qlearning import QLearning
from deep_q_function import DeepQFunction
from tests.plot import Plot

version = "Freeway-ramDeterministic-v4"
mdp = ALEWrapper(version=version)

# Get number of actions and state size from gym action space
action_space = len(mdp.get_actions())
state_space = len(mdp.get_initial_state())

runs = 5
all_rewards = []
for _ in range(runs):

    # Instantiate the critic
    qfunction = DeepQFunction(state_space, action_space, hidden_dim=16)
    learner = QLearning(mdp, qfunction)
    rewards = learner.execute(episodes=30)

    all_rewards.append(rewards)

labels = ["Deep Q learner " + str(i) for i in range(runs)]
Plot.plot_cumulative_rewards(labels, all_rewards, smoothing_factor=0.0)
