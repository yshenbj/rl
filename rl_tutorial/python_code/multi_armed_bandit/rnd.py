import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import math
import random
from multi_armed_bandit.multi_armed_bandit import MultiArmedBandit

class RNDModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(RNDModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.target_fc1 = nn.Linear(input_dim, hidden_dim)
        self.target_fc2 = nn.Linear(hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.fc1.parameters())

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict_target(self, x):
        x = torch.relu(self.target_fc1(x))
        x = self.target_fc2(x)
        return x

    def update_target(self):
        self.target_fc1.load_state_dict(self.fc1.state_dict())
        self.target_fc2.load_state_dict(self.fc2.state_dict())

class RNDBandit(MultiArmedBandit):
    def __init__(self, num_actions, state_dim, rnd_model, epsilon=0.1):
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.rnd_model = rnd_model
        self.epsilon = epsilon

    def select(self, state, actions, qfunction):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            rnd_prediction = self.rnd_model.predict_target(state_tensor).numpy()

        # Calculate novelty scores based on RND prediction
        novelty_scores = np.mean(np.abs(rnd_prediction - state), axis=1)

        # With probability epsilon, explore a random action
        if random.random() < self.epsilon:
            return random.choice(actions)
        else:
            # Choose action with highest novelty score
            return np.argmax(novelty_scores)

    def update(self, action, state):

        # Update RND model
        self.rnd_model.optimizer.zero_grad()
        state_tensor = torch.tensor(state, dtype=torch.float32)
        rnd_prediction = self.rnd_model(state_tensor)
        target_rnd_prediction = self.rnd_model.predict_target(state_tensor)
        rnd_loss = nn.functional.mse_loss(rnd_prediction, target_rnd_prediction)
        rnd_loss.backward()
        self.rnd_model.optimizer.step()
