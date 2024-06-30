import torch
import torch.nn as nn
from value_function import ValueFunction
from torch.optim import Adam


class DeepValueFunction(ValueFunction):
    """
    A neural network to represent the Value-function.
    This class uses PyTorch for the neural network framework (https://pytorch.org/).
    """

    def __init__(
            self, state_space, hidden_dim=64, alpha=0.001
    ):
        # Create a sequential neural network to represent the Q function
        self.value_network = nn.Sequential(
            nn.Linear(in_features=state_space, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=1),
        )
        self.optimiser = Adam(self.value_network.parameters(), lr=alpha)

    def update(self, state, delta):
        self.update_batch([state], [delta])

    def update_batch(self, states, deltas):
        states_tensor = torch.as_tensor(states, dtype=torch.float32)
        deltas_tensor = torch.as_tensor(deltas, dtype=torch.float32)
        
        values = self.value_network(states_tensor)

        loss = nn.functional.smooth_l1_loss(
            values, 
            (values.clone().detach().squeeze(1) + deltas_tensor).unsqueeze(1))

        self.optimiser.zero_grad()
        loss.backward()  # Back-propagate the loss through the network
        self.optimiser.step()  # Do a gradient descent step with the optimiser

    def get_value(self, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        value = self.value_network(state_tensor)
        return value.item()

    def get_values(self, states):
        states_tensor = torch.as_tensor(states, dtype=torch.float32)
        values = self.value_network(states_tensor)
        return values.squeeze(1).tolist()

