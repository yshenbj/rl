# Black Jack environmemt.
import numpy as np 
import random


class BlackJackSingleEnv():
    def __init__(self, print_cards=True):
        self.p = print_cards
        # action space: (Stick, Twist)
        self.act_space = (0, 1)
        # state space: (Dealer's showing card, Current sum, Number of ace)
        self.obs_space = [0, 0, 0] 
        self.hidden_card = None
        self.decks = [2, 3, 4, 5, 6, 7, 8, 9] * 4 + [10] * 12 + [1] * 4
        random.shuffle(self.decks)

    def draw(self, card, ls):
        if self.p:
            print(card)
        # ls: (Current sum, Number of ace)
        if card > 1: 
            ls[0] += card
        else:
            ls[1] += card
        return ls


    def solve_sum(self, ls):
        # ls: (Current sum, Number of ace)
        x = ls[0]
        for _ in range(ls[1]):
            if x + 11 > 21:
                x += 1
            else: 
                x += 11
        return x

    def solve_stick(self):
        if self.p:
            print("Dealer's card:")
        dealer_ls = [0, 0]
        dealer_ls = self.draw(self.obs_space[0], dealer_ls)
        dealer_ls = self.draw(self.hidden_card, dealer_ls)
        dealer_sum = self.solve_sum(dealer_ls) 
        while dealer_sum < 17:
            card = self.decks.pop(0)
            dealer_ls = self.draw(card, dealer_ls) 
            dealer_sum = self.solve_sum(dealer_ls)  
        if dealer_sum > 21:
            return 1 
        else:
            player_sum = self.solve_sum(self.obs_space[1:])
            if player_sum > dealer_sum:
                return 1
            elif player_sum < dealer_sum:
                return -1 
            else:
                return 0

    def play(self, agent):
        # Draw initial cards.
        if self.p: 
            print("Player's card:")
        self.obs_space[1:] = self.draw(self.decks.pop(0), self.obs_space[1:])
        self.obs_space[0] = self.decks.pop(0)
        if self.p: 
            print("Dealer's card:")
            print(self.obs_space[0])     
        if self.p: 
            print("Player's card:")
        self.obs_space[1:] = self.draw(self.decks.pop(0), self.obs_space[1:])
        self.hidden_card = self.decks.pop(0) 
        # Play game.
        while True:
            if self.p:
                print("Your action? (Stay:0 / Hit:1)")
            act = agent.decide(self.obs_space)
            assert act in self.act_space
            if act == self.act_space[0]:
                reward = self.solve_stick()
                break
            elif act == self.act_space[1]:
                if self.p:
                    print("Player's card:")
                self.obs_space[1:] = self.draw(self.decks.pop(0), self.obs_space[1:])
                if self.solve_sum(self.obs_space[1:]) > 21:
                    reward = -1
                    break 
                else:
                    reward = 0
                    continue 
        if self.p:
            if reward > 0:
                print("You win!")
            elif reward < 0:
                print("You lose!")
            else:
                print("It's a draw!")
        return reward
    

class human():
    def decide(self, obs_space):
        # print(obs_space)
        act = int(input())
        return act 
    
    
def main():
    h = human()
    e = BlackJackSingleEnv()
    e.play(h)


if __name__ == "__main__":
    main()