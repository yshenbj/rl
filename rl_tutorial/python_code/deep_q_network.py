import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from qfunction import QFunction


class DeepQFunction(nn.Module, QFunction):

    def __init__(self, state_space, action_space, hidden_dim=128, alpha=1e-4):
        super(DeepQFunction, self).__init__()
        self.layer1 = nn.Linear(state_space, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_space)
        self.optimiser = optim.AdamW(self.parameters(), lr=alpha, amsgrad=True)

    """ A forward pass through the network """

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def get_q_values(self, states, actions):
        states_tensor = torch.as_tensor(states, dtype=torch.float32)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long)
        with torch.no_grad():
            q_values = self.forward(states_tensor).gather(1, actions_tensor.unsqueeze(1))
        return q_values.squeeze(1).tolist()

    def get_q_value(self, state, action):
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        action_tensor = torch.as_tensor(action, dtype=torch.long)

        with torch.no_grad():
            q_values = self.forward(state_tensor)
        q_value = q_values[action]
        return q_value.item()

    def get_max_q_values(self, states):
        states_tensor = torch.as_tensor(states, dtype=torch.float32)
        with torch.no_grad():
            max_q_values = self(states_tensor).max(1).values
        return max_q_values.tolist()

    def get_max_pair(self, state, actions):
        state_tensor = torch.as_tensor(state, dtype=torch.float32)

        with torch.no_grad():
            q_values = self.forward(state_tensor)
        arg_max_q = None
        max_q = float("-inf")
        for action in actions:
            q_value = q_values[action].item()
            if max_q < q_value:
                arg_max_q = action
                max_q = q_value
        return (arg_max_q, max_q)

    def update(self, state, action, delta):
        self.batch_update([state], [action], [delta])

    def optimize_model_old(self, states, actions, next_states, dones, rewards, target_net):

        states_tensor = torch.as_tensor(states, dtype=torch.float32)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long)
        #deltas_tensor = torch.as_tensor(deltas, dtype=torch.float32)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32)

        #next_states_tensor = torch.as_tensor(next_states, dtype=torch.float32)
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              next_states)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in next_states
                                                    if s is not None])
 
        
        # Compute Q-values for current states
        state_action_values = self.forward(states_tensor).gather(
            1, actions_tensor.unsqueeze(1)
        )

        #state_action_values = self.forward(states_tensor) #.gather(1, actions_tensor)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(len(states))
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * 1.0) + rewards_tensor

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimiser.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.parameters(), 100)
        self.optimiser.step()

    def optimize_model(self, states, actions, next_states, dones, rewards, next_state_values):

        states_tensor = torch.as_tensor(states, dtype=torch.float32)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long)
        #deltas_tensor = torch.as_tensor(deltas, dtype=torch.float32)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32)

        #next_states_tensor = torch.as_tensor(next_states, dtype=torch.float32)
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              next_states)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in next_states
                                                    if s is not None])
 
        
        # Compute Q-values for current states
        state_action_values = self.forward(states_tensor).gather(
            1, actions_tensor.unsqueeze(1)
        )

        #state_action_values = self.forward(states_tensor) #.gather(1, actions_tensor)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        #next_state_values = torch.zeros(len(states))
        #with torch.no_grad():
        #    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * 1.0) + rewards_tensor

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimiser.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.parameters(), 100)
        self.optimiser.step()

    def batch_update(self, states, actions, deltas):

        states_tensor = torch.as_tensor(states, dtype=torch.float32)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long)
        deltas_tensor = torch.as_tensor(deltas, dtype=torch.float32)

        # Compute current Q-values for current state-action pairs
        q_values = self.forward(states_tensor).gather(
            1, actions_tensor.unsqueeze(1)
        )

        # Calculate the loss. Deltas already contains the lost, but connect it
        # to q_values via the loss function to enable back propagation
        loss = nn.functional.smooth_l1_loss(
            q_values,
            (q_values.clone().detach().squeeze(1) + deltas_tensor).unsqueeze(1),
        )

        # Optimise the model
        self.optimiser.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.parameters(), 100)
        self.optimiser.step()

    def soft_update(self, policy_qfunction, tau=0.005):
        target_dict = self.state_dict()
        policy_dict = policy_qfunction.state_dict()
        for key in policy_dict:
            target_dict[key] = policy_dict[key] * tau + target_dict[key] * (1 - tau)
        self.load_state_dict(target_dict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
