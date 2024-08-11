'''
import random 
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch.nn.functional as F

from policy import StochasticPolicy

class DeepNeuralNetworkPolicy(nn.Module):

    def __init__(self, state_space, action_space, hidden_dim=64, alpha=0.001):
        super(DeepNeuralNetworkPolicy, self).__init__()
        self.layer1 = nn.Linear(state_space, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_space)
        self.optimiser = optim.AdamW(self.parameters(), lr=alpha, amsgrad=True)
        self.action_space = action_space

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return F.softmax(self.layer3(x), dim=-1)

    def select_action(self, state, actions):
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_logits = self.forward(state_tensor)

        # Mask out the logits of unavailable actions
        mask = torch.full_like(action_logits, float('-inf'))
        mask[actions] = 0
        masked_logits = action_logits + mask

        # Create a categorical distribution over the masked logits
        #print("marked logits = " + str(masked_logits))
        try:
            action_distribution = Categorical(logits=masked_logits)
        except Exception as e:
            print(e)
            print(state)
            print(state_tensor)
            print(actions)
            print(masked_logits)
        #action_distribution = Categorical(logits=action_logits)
        action = action_distribution.sample()
        return action.item() 

    def get_probability(self, state, action):
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.as_tensor(action, dtype=torch.long)
        action_probs = self.forward(state_tensor).squeeze(0)
        return action_probs[action_tensor].item()

    def evaluate_actions(self, states, actions):
        states_tensor = torch.as_tensor(states, dtype=torch.float32)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long)
        action_probs = self.forward(states_tensor)
        action_log_probs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1)))
        return action_log_probs.squeeze(1).view(1, -1)
    
        ##action_logits = self.policy_network(states)
        #action_distribution = Categorical(logits=action_logits)
        #log_prob = action_distribution.log_prob(actions.squeeze(-1))
        #return log_prob.view(1, -1)

    def update(self, states, actions, deltas):

        # Shuffle the list
        indices = list(range(len(states)))

        # Shuffle the indices
        random.shuffle(indices)

        # Use the shuffled indices to reorder the original lists
        states = [states[i] for i in indices]
        actions =  [actions[i] for i in indices]
        deltas = [deltas[i] for i in indices]

        states_tensor = torch.as_tensor(states, dtype=torch.float32)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long)
        deltas_tensor = torch.as_tensor(deltas, dtype=torch.float32)

        # Compute log probabilities of actions taken
        log_probs = self.evaluate_actions(states_tensor, actions_tensor)

        # Calculate the loss
        loss = -(log_probs * deltas_tensor).mean()

        # Optimise the model
        self.optimiser.zero_grad()
        loss.backward()

        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient detected in {name}")

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.parameters(), 100)
        self.optimiser.step()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
'''
'''
# Example usage
# state_space = 4
# action_space = 2
# policy_net = PolicyNetwork(state_space, action_space)
'''
'''
class DeepNeuralNetworkPolicy(nn.Module, StochasticPolicy):

    def __init__(self, state_space, action_space, hidden_dim=64, alpha=0.001):
        super(DeepNeuralNetworkPolicy, self).__init__()
        self.layer1 = nn.Linear(state_space, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_space)
        self.optimiser = optim.AdamW(self.parameters(), lr=alpha, amsgrad=True)
        self.action_space = action_space

        #nn.init.normal_(self.policy_network[0].weight, mean=0.001, std=0.01)
        #nn.init.normal_(self.policy_network[2].weight, mean=0.001, std=0.01)
        #nn.init.normal_(self.policy_network[4].weight, mean=0.001, std=0.01)
        #nn.init.constant_(self.policy_network[0].bias, 0)
        #nn.init.constant_(self.policy_network[2].bias, 0)
        #nn.init.constant_(self.policy_network[4].bias, 0)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return F.softmax(self.layer3(x), dim=-1)
 

    """ Select an action using a forward pass through the network """

    def select_action(self, state, actions):
        #print("\n\nstate = " + str(state))
        #print("actions = " + str(actions))
        # Convert the state into a tensor so it can be passed into the network
        state = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_logits = self.forward(state)
        #print("logits = " + str(action_logits))

        # Mask out the logits of unavailable actions
        mask = torch.full_like(action_logits, float('-inf'))
        mask[actions] = 0
        masked_logits = action_logits + mask

        # Create a categorical distribution over the masked logits
        #print("marked logits = " + str(masked_logits))
        action_distribution = Categorical(logits=masked_logits)
        #action_distribution = Categorical(logits=action_logits)
        action = action_distribution.sample()
        return action.item() 

    """ Get the probability of an action being selected in a state """

    def get_probability(self, state, action):
        state = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_logits = self.forward(state)
        # A softmax layer turns action logits into relative probabilities
        probabilities = F.softmax(input=action_logits, dim=-1).tolist()

        # Convert from a tensor encoding back to the action space
        return probabilities[action] 

    def update(self, states, actions, deltas):
        # Convert to tensors to use in the network
        deltas_tensor = torch.as_tensor(deltas, dtype=torch.float32)
        states_tensor = torch.as_tensor(states, dtype=torch.float32)
        actions_tensor = torch.as_tensor(actions) 

        action_probs = self.forward(states_tensor)
        action_log_probs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1))

        # Construct a loss function, using negative to descend the gradient (not ascend)
        #action_distribution = Categorical(logits=action_probs.squeeze(1))
        #torch.set_printoptions(edgeitems=2048)
        #print(action_log_probs)
        #print(action_distribution.probs)
        #print(actions_tensor)
        #print(deltas_tensor)
        loss = -(action_log_probs * deltas_tensor).mean()
        #print("loss = " + str(loss))
        self.optimiser.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.parameters(), 100)
        self.optimiser.step()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

'''



import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import torch.nn.functional as F

from policy import StochasticPolicy


class DeepNeuralNetworkPolicy(StochasticPolicy):
    """
    An implementation of a policy that uses a PyTorch (https://pytorch.org/) 
    deep neural network to represent the underlying policy.
    """

    def __init__(self, mdp, state_space, action_space, hidden_dim=64, alpha=0.001):
        self.mdp = mdp
        self.state_space = state_space
        self.action_space = action_space

        # Define the policy structure as a sequential neural network.
        self.policy_network = nn.Sequential(
            nn.Linear(in_features=self.state_space, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=self.action_space),
        )

        # The optimiser for the policy network, used to update policy weights
        self.optimiser = Adam(self.policy_network.parameters(), lr=alpha)

        # A two-way mapping from actions to integer IDs for ordinal encoding
        actions = self.mdp.get_actions()
        self.action_to_id = {actions[i]: i for i in range(len(actions))}
        self.id_to_action = {
            action_id: action for action, action_id in self.action_to_id.items()
        }

    """ Select an action using a forward pass through the network """

    def select_action(self, state, actions):
        # Convert the state into a tensor so it can be passed into the network
        state = torch.as_tensor(state, dtype=torch.float32)
        action_logits = self.policy_network(state)
        action_distribution = Categorical(logits=action_logits)
        action = action_distribution.sample()
        return self.id_to_action[action.item()]

    """ Get the probability of an action being selected in a state """

    def get_probability(self, state, action):
        state = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_logits = self.policy_network(state)
        # A softmax layer turns action logits into relative probabilities
        probabilities = F.softmax(input=action_logits, dim=-1).tolist()

        # Convert from a tensor encoding back to the action space
        return probabilities[self.action_to_id[action]]

    def evaluate_actions(self, states, actions):
        action_logits = self.policy_network(states)
        action_distribution = Categorical(logits=action_logits)
        log_prob = action_distribution.log_prob(actions.squeeze(-1))
        return log_prob.view(1, -1)

    def update(self, states, actions, deltas):
        # Convert to tensors to use in the network
        deltas = torch.as_tensor(deltas, dtype=torch.float32)
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor([self.action_to_id[action] for action in actions])

        action_log_probs = self.evaluate_actions(states, actions)

        # Construct a loss function, using negative because we want to descend,
        # not ascend the gradient
        loss = -(action_log_probs * deltas).mean()
        self.optimiser.zero_grad()
        loss.backward()

        # Take a gradient descent step
        self.optimiser.step()

    def save(self, filename):
        torch.save(self.policy_network.state_dict(), filename)

    def load(self, filename):
        self.policy_network.load_state_dict(torch.load(filename))
