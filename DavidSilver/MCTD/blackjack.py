# Black Jack environmemt.
import numpy as np 
import random

class BlackJackSingleEnv():
    def __init__(self, agent):
        self.agent = agent
        self.act_space = (0, 1)
        self.obs_space = (None, None, None)
        self.decks = [2, 3, 4, 5, 6, 7, 8, 9] * 4 + [10] * 12 + ['a'] * 4
        random.shuffle(self.decks)
    
    def step(self, act):
        pass

    def get_reward(self):
        pass