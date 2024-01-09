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
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        canvas = pygame.Surface(
            (self.window_size, self.window_size)
        )
        canvas.fill((255, 255, 255))
        pix_square_size = (self.window_size / 3)
        # pygame.draw.rect(
        #     canvas, 
        #     (255, 0, 0),
        #     pygame.Rect(
        #     )
        # )
        for i in range(3):
            for j in range(3):
                if self._deck[i, j] == -1:
                    pygame.draw.circle(
                        canvas,
                        (255, 255, 255),
                        (pix_square_size * (0.5 + i),  pix_square_size * (0.5 + j))
                    )
                elif self._deck[i, j] == 1:
                    pass
    
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
import numpy as np

deck = np.zeros((3, 3))
ls = deck.reshape(-1).copy()
print(ls)
ls[0] = -1
print(ls)
print(deck)
# %%
import pygame

pygame.init()
window = pygame.display.set_mode(
    (512, 512)
)
clock = pygame.time.Clock()
canvas = pygame.Surface((512, 512))
canvas.fill((255, 255, 255))

window.blit(canvas, canvas.get_rect())
pygame.event.pump()
pygame.display.update()
clock.tick(10)
pygame.display.quit()
pygame.quit()

# %%
# Example file showing a circle moving on screen
import pygame

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
dt = 0

player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("purple")

    pygame.draw.circle(screen, "red", player_pos, 40)

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        player_pos.y -= 300 * dt
    if keys[pygame.K_s]:
        player_pos.y += 300 * dt
    if keys[pygame.K_a]:
        player_pos.x -= 300 * dt
    if keys[pygame.K_d]:
        player_pos.x += 300 * dt

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000

pygame.quit()
# %%
pygame.Vector2(512 / 2, 512 / 2)
# %%
