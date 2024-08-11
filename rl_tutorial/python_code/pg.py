import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state)
        #print(action_log_probs)
        #print(action_distribution.probs)
        #print(actions_tensor)
        #print(deltas_tensor)
        action = torch.multinomial(probs, 1).item()
        return action

    def update(self, states, actions, rewards, optimizer, gamma=0.99):
        R = 0
        policy_loss = []
        returns = []
        
        # Compute the discounted returns
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        for state, action, R in zip(states, actions, returns):
            state = torch.FloatTensor(state).unsqueeze(0)
            probs = self.forward(state)
            log_prob = torch.log(probs.squeeze(0)[action])
            policy_loss.append(-log_prob * R)
        optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()


import gymnasium as gym

def policy_gradient(env, policy_net, optimizer, episodes, gamma=0.99):
    for episode in range(episodes):
        state, _ = env.reset()
        states, actions, rewards = [], [], []
        done = False
        
        while not done:
            
            action = policy_net.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
            
            if done:
                policy_net.update(states, actions, rewards, optimizer, gamma)
                break
        
        if episode % 10 == 0:
            print(f'Episode {episode}, Total Reward: {sum(rewards)}')

# Create the environment
env = gym.make("ALE/Frogger-ram-v5")
#env = gym.make("Freeway-ramDeterministic-v4")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize the policy network and optimizer
policy_net = PolicyNetwork(state_dim, action_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-2)

# Train the policy network
policy_gradient(env, policy_net, optimizer, episodes=1000)
