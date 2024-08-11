from actor_critic import ActorCritic


class QActorCritic(ActorCritic):
    def __init__(self, mdp, actor, critic):
        super().__init__(mdp, actor, critic)

    def update_actor(self, rewards, states, actions, next_states, dones):
        q_values = self.critic.get_q_values(states, actions)
        next_state_q_values = self.critic.get_q_values(next_states, actions)
        deltas = [
            reward + (self.mdp.get_discount_factor() * next_state_q_value) - q_value
            if not done else (reward - q_value)
            for reward, next_state_q_value, q_value, done in zip(
                rewards, next_state_q_values, q_values, dones
            )
        ]

        self.actor.update(states, actions, deltas)

    def update_critic(self, reward, state, action, next_state, done):
        q_value = self.critic.get_q_value(state, action)
        if done:
            delta = reward - q_value
        else:
            actions = self.mdp.get_actions(next_state)
            next_state_value = self.critic.get_max_q(next_state, actions)
            delta = reward + self.mdp.get_discount_factor() * next_state_value - q_value
        self.critic.update(state, action, delta)

    def calculate_deltas(self, rewards):
        """
        Generate a list of the discounted future rewards at each step of an episode
        Note that discounted_reward[T-2] = rewards[T-1] + discounted_reward[T-1] * gamma.
        We can use that pattern to populate the discounted_rewards array.
        """
        T = len(rewards)
        discounted_future_rewards = [0 for _ in range(T)]

        # The final discounted reward is the reward you get at that step
        discounted_future_rewards[T - 1] = rewards[T - 1]
        for t in reversed(range(0, T - 1)):
            discounted_future_rewards[t] = (
                rewards[t]
                + discounted_future_rewards[t + 1] * self.mdp.get_discount_factor()
            )
        deltas = []
        for t in range(len(discounted_future_rewards)):
            deltas += [
                (self.mdp.get_discount_factor() ** t)
                * discounted_future_rewards[t]
            ]
        return deltas