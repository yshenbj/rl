import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces


def is_end(board, mark):
    n_rows, n_cols = board.shape
    for row_index in range(n_rows):
        row = board[row_index, :]
        if (row == mark).all():
            return 1, True
    for col_index in range(n_rows):
        col = board[:, col_index]
        if (col == mark).all():
            return 1, True    
    if (board.diagonal() == mark).all() or (np.fliplr(board).diagonal() == mark).all():
        return 1, True
    else:
        return 0, False


class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode=None):
        self.window_size = 512
        # Observation space is a 3 * 3 deck.
        self.observation_space = spaces.Box(0, 2, shape=(3, 3), dtype=np.int8)
        self.action_space = spaces.Discrete(9)
        self.agent_index_space = spaces.Discrete(2)
        self.agent_mark_mapping = {
            0: 1,
            1: 2
        }
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
        self._agent_index += 1
        if self._agent_index >= self.agent_index_space.n:
            self._agent_index = 0        

        mark = self.agent_mark_mapping[self._agent_index]
        move = (action // 3, action % 3)
        reward, terminated = 0, False
        if self._board[move] == 0:
            self._board[move] = mark
            reward, terminated = is_end(self._board, mark)

        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), reward, terminated, False, self._get_info()
    
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
        canvas.fill((0, 0, 0))
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
                if self._board[i, j] == 1:
                    pygame.draw.circle(
                        canvas,
                        (255, 255, 255),
                        (pix_square_size * (0.5 + j),  pix_square_size * (0.5 + i)),
                        pix_square_size / 2 - 5
                    )
                    pygame.draw.circle(
                        canvas,
                        (0, 0, 0),
                        (pix_square_size * (0.5 + j),  pix_square_size * (0.5 + i)),
                        pix_square_size / 2 - 10
                    )
                elif self._board[i, j] == 2:
                    pygame.draw.line(
                        canvas, 
                        (255, 255, 255),
                        (pix_square_size * j + 5,  pix_square_size * i + 5),
                        (pix_square_size * (j + 1) - 5,  pix_square_size * (i + 1) - 5),
                        7
                    )
                    pygame.draw.line(
                        canvas, 
                        (255, 255, 255),
                        (pix_square_size * (j + 1) - 5,  pix_square_size * i + 5),
                        (pix_square_size * j + 5,  pix_square_size * (i + 1) - 5),
                        7
                    )
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas), axis=(1, 0, 2))
            )
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()