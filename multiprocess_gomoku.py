import copy
import time

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm

import games


class Net(nn.Module):
    def __init__(self, size=13):
        super().__init__()
        self.size = size
        # common layers
        self.conv1 = nn.Conv2d(2, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * size * size, 2 * size * size)
        self.act_fc2 = nn.Linear(2 * size * size, size * size)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.val_fc1 = nn.Linear(4 * size * size, 2 * size * size)
        self.val_fc2 = nn.Linear(2 * size * size, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.size * self.size)
        x_act = F.relu(self.act_fc1(x_act))
        x_act = self.act_fc2(x_act)
        x_act = F.tanh(x_act)
        x_act = F.softmax(x_act, -1)
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 4 * self.size * self.size)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = self.val_fc2(x_val)
        x_val = F.tanh(x_val)

        return x_act, x_val
    
class PolicyValueNet:
    def __init__(self, lr=1e-3, c=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Net().to(self.device)
        self.lr = lr
        self.c = c
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.c)
    
    def policy_value(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action_p, value = self.net(state)
        return action_p.view(-1).cpu().numpy(), value.item()
    
    def update(self, state, mcts_p, reward):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        mcts_p = torch.tensor(mcts_p, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.optimizer.zero_grad()
        action_p, value = self.net(state)
        # cross entorpy loss for the search probabilities 
        policy_loss = torch.mean(torch.sum(mcts_p * torch.log(action_p), -1))
        # mse loss for the state value
        value_loss = F.mse_loss(value.view(-1), reward)
        # total loss
        loss = value_loss - policy_loss
        loss.backward()
        self.optimizer.step()
        return -policy_loss.item(), value_loss.item()
            
    def save(self, filename):
        torch.save(self.net.state_dict(), filename)
    
    @classmethod
    def load(cls, filename):
        policy_value_net = cls()
        policy_value_net.net.load_state_dict(torch.load(filename))
        return policy_value_net


class AgentNode:
    """
    Agent node of MCTS
    """
    def __init__(self, parent, action, num_actions, P, N=0, W=0):
        """
        Each state_action pair (s, a) stores a set of statistics, {N(s, a), W(s, a), Q(s, a), P(s, a)},
        where N(s, a) is the visit count, W(s, a) is the total action-value, Q(s, a) is the mean action-value,
        and P(s, a) is the prior probability of selecting a in s.
        """
        self.parent = parent
        self.action = action
        self.num_actions = num_actions
        self.P = P
        self.N = N
        self.W = W
        self.children = {}
        self.child_N = np.zeros(num_actions, dtype=np.float32)
        self.child_W = np.zeros(num_actions, dtype=np.float32)
        self.child_P = None
        self.agent_index = None
        self.is_expanded = False
    
    def select(self, c_puct_base, c_puct_init, action_mask):
        c_puct = np.log((1 + self.N + c_puct_base) / c_puct_base) + c_puct_init
        Q = self.child_W / np.where(self.child_N > 0, self.child_N, 1)
        U = c_puct * self.child_P * np.sqrt(self.N) / (1 + self.child_N)
        UCB = U - Q
        action = np.where(action_mask, UCB, float('-inf')).argmax()
    
        if action not in self.children.keys():
            self.children[action] = AgentNode(
                self,
                action,
                self.num_actions,
                self.child_P[action]
            )

        return action, self.children[action]
    
    def expand(self, agent_index, next_P):
        self.agent_index = agent_index
        self.child_P = next_P
        self.is_expanded = True

    def back_propagate(self, value):
        self.N += 1
        self.W += value
        if self.parent:
            self.parent.child_N[self.action] = self.N
            self.parent.child_W[self.action] = self.W
            self.parent.back_propagate(-value)
            
    def as_root(self):
        self.parent = None
        self.action = None
        self.P = 1
    
    @property
    def Q(self):
        return self.W / self.N


class MCTSPlayer:
    def __init__(
            self, 
            policy_value_net, 
            c_puct_base=100, 
            c_puct_init=1, 
            num_simulations=1000, 
            noise=True, 
            deterministic=False
        ):
        self.policy_value_net = policy_value_net
        self.c_puct_base = c_puct_base
        self.c_puct_init = c_puct_init
        self.num_simulations = num_simulations
        self.noise = noise
        self.deterministic = deterministic
        self.rng = np.random.default_rng()
    
    def to_state(self, observation, info, agent_mark_mapping):
        """ 
        Transfer environmental observation and information to a state tensor as neural network input.
        Board observation will transfer to an N * N * M image stack, each plane represent the board positions 
        oriented to a certain player (N * N with dummy), current player's plane is on the top.
        state: numpy array with shape (2, 3, 3)
        """
        agent_index = info['agent_index']
        mark_list = list(agent_mark_mapping.values())
        num_agents = len(mark_list)
        array_list = []

        for i in range(num_agents):
            index = (agent_index + i) % num_agents
            mark = mark_list[index]
            array_list.append(observation == mark)
            
        state = np.stack(array_list, dtype=np.float32)
        
        return state
    
    def add_dirchleet_noise(self, node, action_mask, epsilon=0.25, alpha=0.03):
        alphas = action_mask * alpha
        noise = self.rng.dirichlet(alphas)
        node.child_P = node.child_P * (1 - epsilon) + noise * epsilon
        
    def get_mcts_p(self, child_N, temperature=1):
        child_N = child_N ** (1 / temperature)
        return child_N / sum(child_N)
    
    def mcts(self, env, observation, info, root_node=None):
        # Initialize environment and root node.
        num_actions = env.unwrapped.action_space.n
        agent_mark_mapping = env.unwrapped.agent_mark_mapping
        root_state = self.to_state(observation, info, agent_mark_mapping)
        root_action_mask = np.equal(observation.flatten(), 0)
        prior_p, value = self.policy_value_net.policy_value(root_state)

        if not root_node:
            root_node = AgentNode(None, None, num_actions, 1)
        root_node.expand(info['agent_index'], prior_p)
        root_node.back_propagate(value)
        
        # Add dirchleet noise.
        if self.noise:
            self.add_dirchleet_noise(root_node, root_action_mask)

        # Start mcts simulation.
        while root_node.N < self.num_simulations:
            sim_env = copy.deepcopy(env)
            node = root_node
            action_mask = root_action_mask
            
            done = False
            while node.is_expanded:
                # SELECT
                action, node = node.select(self.c_puct_base, self.c_puct_init, action_mask)
                # INTERACT
                observation, reward, terminated, truncated, info = sim_env.step(action)
                action_mask = np.equal(observation.flatten(), 0)
                done = terminated or truncated
                if done:                  
                    break
            
            if done:
                # BACK PROPAGATE (REWARD)
                node.back_propagate(-reward)
            else:
                # EVALUATE
                state = self.to_state(observation, info, agent_mark_mapping)                    
                prior_p, value = self.policy_value_net.policy_value(state)
                # EXPAND
                node.expand(info['agent_index'], prior_p)
                # BACK PROPAGATE (VALUE)
                node.back_propagate(value)
        
        # Choose best action for root node (deterministic or stochastic).
        mcts_p = self.get_mcts_p(root_node.child_N)
        if self.deterministic:
            action = np.argmax(mcts_p)
        else:
            action = self.rng.choice(np.arange(num_actions), p=mcts_p)
        
        if action in root_node.children.keys():
            next_root_node = root_node.children[action]
            next_root_node.as_root()
        else:
            next_root_node = None

        return root_state, action, mcts_p, next_root_node


def selfplay(env, policy_value_net, argumentation=True):
    agent_index_list = []
    state_list = []
    mcts_p_list = []
    
    observation, info = env.reset()
    player = MCTSPlayer(policy_value_net, noise=True, deterministic=False)
    root_node = None
    done = False
    
    while not done:
        agent_index_list.append(info['agent_index'])
        root_state, action, mcts_p, root_node = player.mcts(env, observation, info, root_node)
        state_list.append(root_state)
        mcts_p_list.append(mcts_p)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    reward_list = [
        reward if index == agent_index_list[-1] else -reward for index in agent_index_list
    ]
    
    if argumentation:
        state_list = state_list \
            + [np.stack([np.fliplr(state[0]), np.fliplr(state[1])]) for state in state_list] \
            + [np.stack([np.flipud(state[0]), np.flipud(state[0])]) for state in state_list] \
            + [np.stack([np.rot90(state[0], 1), np.rot90(state[0], 1)]) for state in state_list] \
            + [np.stack([np.rot90(state[0], 2), np.rot90(state[0], 2)]) for state in state_list] \
            + [np.stack([np.rot90(state[0], 3), np.rot90(state[0], 3)]) for state in state_list] \
            + [np.stack([np.transpose(state[0]), np.transpose(state[0])]) for state in state_list] \
            + [np.stack([np.transpose(state[0, ::-1, ::-1]), np.transpose(state[0, ::-1, ::-1])]) for state in state_list]

        mcts_p_list = mcts_p_list \
            + [np.fliplr(np.reshape(mcts_p, (13, 13))).flatten() for mcts_p in mcts_p_list] \
            + [np.flipud(np.reshape(mcts_p, (13, 13))).flatten() for mcts_p in mcts_p_list] \
            + [np.rot90(np.reshape(mcts_p, (13, 13)), 1).flatten() for mcts_p in mcts_p_list] \
            + [np.rot90(np.reshape(mcts_p, (13, 13)), 2).flatten() for mcts_p in mcts_p_list] \
            + [np.rot90(np.reshape(mcts_p, (13, 13)), 3).flatten() for mcts_p in mcts_p_list] \
            + [np.transpose(np.reshape(mcts_p, (13, 13))).flatten() for mcts_p in mcts_p_list] \
            + [np.transpose(np.reshape(mcts_p, (13, 13))[::-1, ::-1]).flatten() for mcts_p in mcts_p_list]
        
        reward_list = reward_list * 8

    return np.array(state_list), np.array(mcts_p_list), np.array(reward_list)


def run(env, policy_value_net, epoch, num_epochs, lock):
    while True:
        if epoch.value < num_epochs:
            state, mcts_p, reward = selfplay(env, policy_value_net)
            lock.acquire()

            policy_value_net.net.train()
            policy_loss, value_loss = policy_value_net.update(state, mcts_p, reward)
            policy_value_net.net.eval()

            epoch.value += 1

            if epoch.value == num_epochs // 2:
                policy_value_net.lr = 1e-4

            if epoch.value % 500 == 0:
                policy_value_net.save(f'gomoku_weights/epoch_{epoch.value}.pth')

            lock.release()
            # print(f'Epoch: {epoch.value} | Policy loss {policy_loss} | Value loss {value_loss}')
        else:
            return


def main(num_epochs=20000, num_parallels=12):
    mp.set_start_method('spawn', force=True)
    env = gym.make('games/Gomoku', max_episode_steps=169)
    policy_value_net = PolicyValueNet(lr=1e-3)
    policy_value_net.net.share_memory()
    epoch = mp.Value('i', 0)
    lock = mp.Lock()
    progress_bar = tqdm(total=num_epochs)

    jobs = []
    for _ in range(num_parallels):
        p = mp.Process(target=run, args=(env, policy_value_net, epoch, num_epochs, lock))
        p.start()
        jobs.append(p)

    tmp_epoch = epoch.value
    while epoch.value < num_epochs:
        progress_bar.update(epoch.value - tmp_epoch)
        time.sleep(1) 

    for p in jobs:
        p.join()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f'Time cost: {round((end_time - start_time) / 60, 2)} min')