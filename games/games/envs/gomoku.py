import numpy as np
import gymnasium as gym
from gymnasium import spaces


def check(ls, mark):
    cnt = 0
    for element in ls:
        if element == mark:
            cnt += 1
        else:
            cnt = 0
    return cnt > 4


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
        self._board = np.zeros((self.size, self.size), dtype=np.int8)
        # Randomly pick an agent.
        self._agent_index = self.agent_index_space.sample()

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        assert self.action_space.contains(action)     
        mark = self.agent_mark_mapping[self._agent_index]
        move = (action // self.size, action % self.size)
        reward, terminated = 0, False
        if self._board[move] == 0:
            self._board[move] = mark
            reward, terminated = self.is_end(mark, move)
            
        self._agent_index += 1
        if self._agent_index >= self.agent_index_space.n:
            self._agent_index = 0 
            
        return self._get_obs(), reward, terminated, False, self._get_info()
    
    def is_end(self, mark, move):
        x, y = move
        row_end = check([self._board[i, y] for i in range(max(0, x - 4), min(self.size, x + 5))], mark)
        col_end = check([self._board[x, i] for i in range(max(0, y - 4), min(self.size, y + 5))], mark)
        diag_end = check([self._board[x+i][y+i] for i in range(max(-x, -y, -4), min(self.size - x, self.size - y, 5))], mark)
        anti_diag_end = check([self._board[x+i][y-i] for i in range(max(-x, y - self.size + 1, -4), min(self.size - x, y + 1, 5))], mark)
        if row_end or col_end or diag_end or anti_diag_end:
            return 1, True
        else:
            return 0, False