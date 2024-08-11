import random

from collections import namedtuple, deque
from itertools import count

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)

class ReplayMemory:
    def __init__(self, memory_size=10000):
        self.memory = deque([], maxlen=memory_size)

    def push(self, state, action, next_state, reward, done):
        self.memory.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ExperienceReplayLearner:
    def __init__(
        self,
        mdp,
        bandit,
        policy_qfunction,
        target_qfunction,
        memory=ReplayMemory(),
        batch_size=128,
        memory_size=10000,
        update_period=1,
    ):
        self.mdp = mdp
        self.bandit = bandit
        self.policy_qfunction = policy_qfunction
        self.target_qfunction = target_qfunction
        self.batch_size = batch_size
        self.memory = memory
        self.update_period = update_period
        self.target_qfunction.soft_update(self.policy_qfunction)

    def execute(self, episodes=100, max_episode_length=float('inf')):

        episode_rewards = []
        for episode in range(episodes):
            state = self.mdp.get_initial_state()
            actions = self.mdp.get_actions(state)
            action = self.bandit.select(state, actions, self.policy_qfunction)
            episode_reward = 0
            for step in count():
                (next_state, reward, done) = self.mdp.execute(state, action)
                actions = self.mdp.get_actions(next_state)
                next_action = self.bandit.select(
                    next_state, actions, self.policy_qfunction
                )

                '''
                if done:
                    next_state2 = None
                    #next_state2 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                    self.memory.push(state, action, next_state2, reward, done)
                else:
                    
                    import torch
                    next_state2 = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                    self.memory.push(state, action, next_state2, reward, done)
                '''
                self.memory.push(state, action, next_state, reward, done)

                # Perform an update on the policy qfunction using a batch
                if len(self.memory) >= self.batch_size and step % self.update_period == 0:
                    transitions = self.memory.sample(self.batch_size)
                    batch = Transition(*zip(*transitions))

                    
                    deltas = self.get_deltas(
                        batch.reward,
                        batch.state,
                        batch.action,
                        batch.next_state,
                        batch.done,
                    )

                    self.policy_qfunction.batch_update(batch.state, batch.action, deltas)

                    '''

                    non_final_mask = torch.tensor(tuple(map(lambda s: s is False,
                                              batch.done)), dtype=torch.bool)
                    non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
                    #print(len(non_final_next_states))
                    states_tensor = torch.as_tensor(batch.state, dtype=torch.float32)
                    actions_tensor = torch.as_tensor(batch.action, dtype=torch.long)
                    #deltas_tensor = torch.as_tensor(deltas, dtype=torch.float32)
                    rewards_tensor = torch.as_tensor(batch.reward, dtype=torch.float32)
                    #sys.exit()
                    next_state_values = torch.zeros(len(states_tensor))
                    with torch.no_grad():
                        next_state_values[non_final_mask] = self.target_qfunction(non_final_next_states).max(1).values
                    
                    self.policy_qfunction.optimize_model(batch.state, batch.action, batch.next_state, batch.done, batch.reward, next_state_values)
                    '''
                    

                # Soft update of the target network's weights
                self.target_qfunction.soft_update(self.policy_qfunction)

                # Move to the next state
                state = next_state
                action = next_action
                episode_reward += reward * (self.mdp.discount_factor ** step)

                if done or step == max_episode_length:
                    break

            episode_rewards.append(episode_reward)

        return episode_rewards

    """ Calculate the deltas for the update """

    def get_deltas(self, rewards, states, actions, next_states, dones):
        q_values = self.policy_qfunction.get_q_values(states, actions)
        next_state_q_values = self.state_values(next_states, actions)
        deltas = [
            reward + (self.mdp.get_discount_factor() * next_state_q_value) - q_value
            if not done else (reward - q_value)
            for reward, next_state_q_value, q_value, done in zip(
                rewards, next_state_q_values, q_values, dones
            )
        ]
        return deltas

    def state_values(self, states, actions):
        return self.target_qfunction.get_max_q_values(states)
