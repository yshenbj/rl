import pygame
import numpy as np
from gymnasium import Env
from gymnasium.space import Dict, Box, Discrete

# %%
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
            self.window = pygame.display.set_mode
    
    def reset(self):
        self._deck = np.zeros((3, 3))
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, info
    
    
       
        
        
# %%
import numpy as np

print(np.zeros((3, 3)))
# %%
