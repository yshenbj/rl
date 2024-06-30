from model_free_learner import ModelFreeLearner

class TemporalDifferenceLearner(ModelFreeLearner):
    def __init__(self, mdp, bandit, qfunction):
        self.mdp = mdp
        self.bandit = bandit
        self.qfunction = qfunction

    def execute(self, episodes=2000):

        rewards = []
        for episode in range(episodes):
            state = self.mdp.get_initial_state()
            actions = self.mdp.get_actions(state)
            action = self.bandit.select(state, actions, self.qfunction)

            episode_reward = 0.0
            step = 0
            while not self.mdp.is_terminal(state):
                (next_state, reward, done) = self.mdp.execute(state, action)
                actions = self.mdp.get_actions(next_state)
                next_action = self.bandit.select(next_state, actions, self.qfunction)

                delta = self.get_delta(reward, state, action, next_state, next_action)
                self.qfunction.update(state, action, delta)

                state = next_state
                action = next_action
                episode_reward += reward * (self.mdp.discount_factor ** step)
                step += 1

            rewards.append(episode_reward)

        return rewards

    """ Calculate the delta for the update """

    def get_delta(self, reward, state, action, next_state, next_action):
        q_value = self.qfunction.get_q_value(state, action)
        next_state_value = self.state_value(next_state, next_action)
        delta = reward + self.mdp.discount_factor * next_state_value - q_value
        return delta

    """ Get the value of a state """

    def state_value(self, state, action):
        abstract
