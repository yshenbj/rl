# Black Jack environmemt.
import numpy as np 
import random

class BlackJackSingleEnv():
    def __init__(self, agent):
        self.agent = agent
        # action space: (Stick, Twist)
        self.act_space = (0, 1)
        # state space: (Dealer's showing card, Current sum, Number of ace)
        self.obs_space = [0, 0, 0] 
        self.hidden_card = None
        self.decks = [2, 3, 4, 5, 6, 7, 8, 9] * 4 + [10] * 12 + [1] * 4
        random.shuffle(self.decks)

    def palyer_draw(self, card):
        if card > 1: 
            self.obs_space[1] += card
        else:
            self.obs_space[2] += card


    def solve_sum(self, ls):
        # ls: (Current sum, Number of ace)
        x = sum(ls[0]) 
        for _ in range(ls[1]):
            if x + 11 > 21:
                x += 1
            else: 
                x += 11
        return x


    def solve_stick(self):
        # dealer_sum = 
        pass


    def play(self):
        # Draw initial cards.
        self.palyer_draw(self.decks.pop(0))
        self.obs_space[0] = self.decks.pop(0)
        self.palyer_draw(self.decks.pop(0))
        self.hidden_card = self.decks.pop(0) 
        # Play game.
        while True:
            act = self.agent.decide(self.obs_space)
            assert act in self.act_space 
            if act == self.act_space[0]:
                reward = self.solve_stick()
                break
            elif act == self.act_space[1]:
                self.obs_space[1] += self.decks.pop(0)
                reward = 0 if self.solve_sum(self.obs_space[1:]) > 21 else 0
                continue 
        return reward