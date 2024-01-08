import pygame
import numpy as np
from gymnasium import Env
from gymnasium.space import Dict, Box, Discrete


class TicTacToeEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode=None):
        self.window_size = 512
        # Observation space is a 3 * 3 deck.
        self.observation_space = Dict(
            {
                "deck": Box(-1, 1, shape=(3, 3), dtype=int)
            }
        )
        # Action can be (x, y) which put X or O. 
        self.action_space = Discrete(4)
        self.render_mode = render_mode
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.     
        """
        self.window = None
        self.clock = None
        
    def _get_obs(self):
        return self._deck
    
    def _get_info(self):
        return 
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.dispaly.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        canvas = pygame.Surface(
            (self.win)
        )
    
    def step(self, player, location):
        if self._deck[location] == 0:
            self._deck[location] = player
        val = player * 3
        terminated = (deck.sum(axis=1) == val).any() or (deck.sum(axis=0) == val).any() or (deck.diagonal().sum() == val) or (np.fliplr(deck).diagonal().sum() == val)
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, terminated, False, info

    def reset(self):
        self._deck = np.zeros((3, 3))
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, info 
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    
       
        
        
# %%
import numpy as np

deck = np.zeros((3, 3))
deck[0, 0] = 1
deck[2, 0] = 1
print(deck)
(deck.sum(axis=1) == -3).any()
(deck.sum(axis=0) == 3).any()
print(np.fliplr(deck).diagonal().sum() == 1)
# %%
