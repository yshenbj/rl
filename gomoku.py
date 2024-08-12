import numpy as np
import gymnasium as gym
from gymnasium import spaces


def is_end(board, mark):
    n_rows, n_cols = board.shape
    for row_index in range(n_rows):
        row = board[row_index, :]
        if (row == mark).all():
            return 1, True
    for col_index in range(n_cols):
        col = board[:, col_index]
        if (col == mark).all():
            return 1, True    
    if (board.diagonal() == mark).all() or (np.fliplr(board).diagonal() == mark).all():
        return 1, True
    else:
        return 0, False


class GomokuEnv(gym.Env):
    
    def __init__(self, size=13):
        self.size = size
        self.observation_space = spaces.Box(0, 2, shape=(size, size), dtype=np.int8)
        self.action_space = spaces.Discrete(size * size)
        self.agent_index_space = spaces.Discrete(2)
        self.agent_mark_mapping = {
            0: 1,
            1: 2
        }
        
    def _get_obs(self):
        return self._board
    
    def _get_info(self):
        # Return the index of agent which ready to act.
        return {"agent_index": self._agent_index}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize the deck with zeros and the agent index.
        self._board = np.zeros((3, 3), dtype=np.int8)
        # Randomly pick an agent.
        self._agent_index = self.agent_index_space.sample()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def step(self, action):
        assert self.action_space.contains(action)     
        mark = self.agent_mark_mapping[self._agent_index]
        move = (action // self.size, action % self.size)
        reward, terminated = 0, False
        if self._board[move] == 0:
            self._board[move] = mark
            reward, terminated = is_end(self._board, mark)
            
        self._agent_index += 1
        if self._agent_index >= self.agent_index_space.n:
            self._agent_index = 0 
        
        if self.render_mode == "human":
            self._render_frame()  
            
        return self._get_obs(), reward, terminated, False, self._get_info()