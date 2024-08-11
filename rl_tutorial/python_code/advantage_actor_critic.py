from actor_critic import ActorCritic

class AdvantageActorCritic(ActorCritic):

    def __init__(self, mdp, actor, critic):
        super().__init__(mdp, actor, critic)


    def update_actor(self, rewards, states, actions, next_states, dones):
        
        values = self.critic.get_values(states)
        next_state_values = self.critic.get_values(next_states)
        '''
        import random
        if random.random() < 0.02:
            for i in range(len(states)):
                print(states[i] + " V(" + str(values[i]) + ") --" + str(actions[i]) + "--> " + str(next_states[i]) + "V(" + str(next_state_values[i]) + ")")
            print("=====\n")
        '''
        advantages = [
            reward + (self.mdp.get_discount_factor() * next_state_value) - value
            if not done else (reward - value)
            for reward, next_state_value, value, done in zip(
                rewards, next_state_values, values, dones
            )
        ]
        
        self.actor.update(states, actions, advantages)

    def update_critic(self, reward, state, action, next_state, done):
        state_value = self.critic.get_value(state)
        next_state_value = self.critic.get_value(next_state)
        delta = reward + self.mdp.get_discount_factor() * next_state_value - state_value
        self.critic.update(state, delta)
