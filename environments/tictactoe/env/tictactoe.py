import pygame
import random
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete, Dict


class TicTacToeEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode=None):
        self.window_size = 512
        # Observation space is a 3 * 3 deck.
        self.observation_space = Box(-1, 1, shape=(3, 3), dtype=int)
        # Action can be (x, y) which put X or O. 
        self.action_space = Discrete(9)
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
        return {"player": self._player}
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        canvas = pygame.Surface(
            (self.window_size, self.window_size)
        )
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / 3
        # Draw lines to separate boxes.
        pygame.draw.aaline(
            canvas, 
            (255, 255, 255),
            (pix_square_size * 1,  pix_square_size * 0),
            (pix_square_size * 1,  pix_square_size * 3)
        )
        pygame.draw.aaline(
            canvas, 
            (255, 255, 255),
            (pix_square_size * 2,  pix_square_size * 0),
            (pix_square_size * 2,  pix_square_size * 3)
        )
        pygame.draw.aaline(
            canvas, 
            (255, 255, 255),
            (pix_square_size * 0,  pix_square_size * 1),
            (pix_square_size * 3,  pix_square_size * 1)
        )
        pygame.draw.aaline(
            canvas, 
            (255, 255, 255),
            (pix_square_size * 0,  pix_square_size * 2),
            (pix_square_size * 3,  pix_square_size * 2)
        )
        # Draw "X"s or "O"s.
        for i in range(3):
            for j in range(3):
                if self._deck[i, j] == -1:
                    pygame.draw.circle(
                        canvas,
                        (255, 255, 255),
                        (pix_square_size * (0.5 + i),  pix_square_size * (0.5 + j)),
                        pix_square_size - 5
                    )
                    pygame.draw.circle(
                        canvas,
                        (0, 0, 0),
                        (pix_square_size * (0.5 + i),  pix_square_size * (0.5 + j)),
                        pix_square_size - 10
                    )
                elif self._deck[i, j] == 1:
                    pygame.draw.line(
                        canvas, 
                        (255, 255, 255),
                        (pix_square_size * i + 5,  pix_square_size * j + 5),
                        (pix_square_size * (i + 1) - 5,  pix_square_size * (j + 1) - 5),
                        7
                    )
                    pygame.draw.line(
                        canvas, 
                        (255, 255, 255),
                        (pix_square_size * (i + 1) - 5,  pix_square_size * j + 5),
                        (pix_square_size * i + 5,  pix_square_size * (j + 1) - 5),
                        7
                    )
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surface.pixels3d(canvas), axis=(1, 0, 2))
            )
    
    def step(self, action):
        location = (action // 3, action % 3)
        if self._deck[location] == 0:
            self._deck[location] = self._player
        val = self._player * 3
        terminated = (self._deck.sum(axis=1) == val).any() or (self._deck.sum(axis=0) == val).any() or (self._deck.diagonal().sum() == val) or (np.fliplr(self._deck).diagonal().sum() == val)
        reward = 1 if terminated else 0
        self._player *= -1
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, terminated, False, info

    def reset(self):
        self._deck = np.zeros((3, 3))
        self._player = random.choice((-1, 1))
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
