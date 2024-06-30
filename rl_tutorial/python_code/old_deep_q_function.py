import torch
import torch.nn as nn
from qfunction import QFunction
from torch.optim import Adam


class DeepQFunction(QFunction):
    """A neural network to represent the Q-function.
    This class uses PyTorch for the neural network framework (https://pytorch.org/).
    """

    def __init__(self, state_space, action_space, hidden_dim=64, alpha=0.001) -> None:

        # Create a sequential neural network to represent the Q function
        self.q_network = nn.Sequential(
            nn.Linear(in_features=state_space, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=action_space),
        )
        self.optimiser = Adam(self.q_network.parameters(), lr=alpha, amsgrad=True)

    def update(self, state, action, delta):
        self.batch_update([state], [action], [delta])

    def batch_update(self, experiences):
        (states, actions, deltas, dones) = zip(*experiences)
        self.batch_update(states, actions, deltas)

    def batch_update(self, states, actions, deltas):
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        deltas_tensor = torch.tensor(deltas, dtype=torch.float32)

        q_values = self.q_network(states_tensor).gather(
            dim=1, index=actions_tensor.unsqueeze(1)
        )

        loss = nn.functional.smooth_l1_loss(
            q_values,
            (q_values.clone().detach().squeeze(1) + deltas_tensor).unsqueeze(1),
        )
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimiser.step()

    def get_q_values(self, states, actions):
        states_tensor = torch.as_tensor(states, dtype=torch.float32)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long)
        with torch.no_grad():
            q_values = self.q_network(states_tensor).gather(
                1, actions_tensor.unsqueeze(1)
            )
        return q_values.squeeze(1).tolist()

    def get_max_q_values(self, states):
        states_tensor = torch.as_tensor(states, dtype=torch.float32)
        with torch.no_grad():
            max_q_values = self.q_network(states_tensor).max(1).values
        return max_q_values.tolist()

    def get_q_value(self, state, action):
        # Convert the state into a tensor
        state = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.q_network(state)

        q_value = q_values[action].item()

        return q_value

    def get_max_pair(self, state, actions):
        # Convert the state into a tensor
        state = torch.as_tensor(state, dtype=torch.float32)

        # Since we have a multi-headed q-function, we only need to pass through the network once
        # call torch.no_grad() to avoid tracking the gradients for this network forward pass
        with torch.no_grad():
            q_values = self.q_network(state)
        arg_max_q = None
        max_q = float("-inf")
        for action in actions:
            q_value = q_values[action].item()
            if max_q < q_value:
                arg_max_q = action
                max_q = q_value
        return (arg_max_q, max_q)

    def soft_update(self, policy_qfunction, tau=0.005):
        target_dict = self.q_network.state_dict()
        policy_dict = policy_qfunction.q_network.state_dict()
        for key in policy_dict:
            target_dict[key] = policy_dict[key] * tau + target_dict[key] * (1 - tau)
        self.q_network.load_state_dict(target_dict)

    def save(self, filename):
        torch.save(self.q_network.state_dict(), filename)

    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename))
