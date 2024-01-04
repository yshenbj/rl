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
        # Action can be (x, y) which put cross or 
        self.action_space = Discrete(4)
        
        
# %%
