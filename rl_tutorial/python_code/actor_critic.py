from itertools import count

from model_free_learner import ModelFreeLearner

class ActorCritic(ModelFreeLearner):
    def __init__(self, mdp, actor, critic):
        self.mdp = mdp
        self.actor = actor  # Actor (policy based) to select actions
        self.critic = critic  # Critic (value based) to evaluate actions

    def execute(self, episodes=100, max_episode_length=float('inf')):
        episode_rewards = []
        for episode in range(episodes):
            actions = []
            states = []
            rewards = []
            next_states = []
            dones = []

            episode_reward = 0
            state = self.mdp.get_initial_state()
            for step in count():
                #action = self.actor.select_action(state, self.mdp.get_actions(state))
                import random
                length = len(self.mdp.get_actions(state))
                action = random.sample(self.mdp.get_actions(state), length)[0]
                (next_state, reward, done) = self.mdp.execute(state, action)
                self.update_critic(reward, state, action, next_state, done)

                # Store the information from this step of the trajectory
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                state = next_state
                episode_reward += reward * (self.mdp.discount_factor ** step)

                if done or step == max_episode_length:
                    break

            self.update_actor(rewards, states, actions, next_states, dones)

            episode_rewards.append(episode_reward)

        return episode_rewards

    """ Update the actor using a batch of rewards, states, actions, and next states """

    def update_actor(self, rewards, states, actions, next_states, dones):
        abstract

    """ Update the critc using a reward, state, action, and next state """

    def update_critic(self, reward, state, action, next_state):
        abstract
