
import sys
import math
import seaborn as sns

import numpy as np
from matplotlib import pyplot as plt


class env():
    """
    Jack's Car Rental example in leacture 3.
    States: Two locations, maximum ofn20 car at each.
    Actions: Move up to 5 cars between location overnight.
    Reward: $10 for each car rented (must be avaliable).
    Transitions: Cars returned and requested randomly.
        Poisson distribution, n returns/requests with prob Poisson(lambda)
        1st location: average requests = 3, average returns = 3.
        2nd loaction: average requests = 4, average returns = 2.
    """
    def __init__(self):
        self.p = np.zeros((21, 21)) 
        self.v = np.zeros((21, 21))
        self.avg_lambda = {"loc1": {"req": 3, "ret": 3}, "loc2": {"req": 4, "ret": 2}}
        self.a = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

    
    def get_action(self, s1, s2, action):
        sign = np.sign(action)
        if sign >= 0:
            action = np.min([20-s1, s2, action])
        else:
            action = np.min([s1, 20-s2, np.abs(action)]) * -1
        return action


    def calculate_transition_matrix_item(self, loc, s_start, s_end):
        """ Calculate P(s=s_start, s=s_end) """
    
        p_start_end = 0
        reward_start_end = 0
        for n_ret in range(max(0, s_end-s_start), 21-s_start):
             n_req = n_ret + s_start - s_end
             p_ret = self.avg_lambda[loc]["ret"] ** n_ret * np.exp(-self.avg_lambda[loc]["ret"]) / math.factorial(n_ret)
             p_req = self.avg_lambda[loc]["req"] ** n_req * np.exp(-self.avg_lambda[loc]["req"]) / math.factorial(n_req)
             #print(s_start, "+", n_ret, "-", n_ret+s_start-s_end, "=", s_end, "prob=", p_ret*p_req)
             p_start_end += p_ret * p_req
             reward_start_end += 10 * n_req * p_ret * p_req
        return p_start_end, reward_start_end


    def calculate_transition_matrix(self, loc):
        # P(ss')
        p_transition_matrix = np.zeros((21, 21)) 
        # P(ss') * R(s'|s)
        reward_transition_matrix = np.zeros((21, 21)) 
        #print(len(p_transition_matrix), len(p_transition_matrix[0]))

        for s_start in range(21):
            for s_end in range(21):
                p_transition_matrix[s_start][s_end] = self.calculate_transition_matrix_item(loc, s_start, s_end)[0]
                reward_transition_matrix[s_start][s_end] = self.calculate_transition_matrix_item(loc, s_start, s_end)[1]

        # P(s'|s) = P(ss') / P(s)
        p_start = np.sum(p_transition_matrix, axis=0)
        p_transition_matrix = p_transition_matrix / p_start
        # R(s'|s) = R(ss') / p(s)
        reward_transition_matrix = reward_transition_matrix / p_start
        
        return p_transition_matrix, reward_transition_matrix


    def get_one_step_value(self, s1_start, s2_start, p_transition_matrix_loc1, p_transition_matrix_loc2):
        # sum(Pss' * v(s') for all s' in S)
        one_step_value = np.sum(np.outer(p_transition_matrix_loc1[s1_start], p_transition_matrix_loc2[s2_start]) * self.v.T)
        return one_step_value
        
        
    def iteration(self, n_iter):
        p_transition_matrix_loc1, reward_transition_matrix_loc1 = self.calculate_transition_matrix("loc1")
        p_transition_matrix_loc2, reward_transition_matrix_loc2 = self.calculate_transition_matrix("loc2")
        #print(p_transition_matrix, reward_transition_matrix)
        
        for _ in range(n_iter):
            # Initialize action-state value function
            q =  np.empty((21, 21, 11))
            for i in range(21):
                for j in range(21):
                    s1, s2 = i, j
                    # Act by policy
                    action_true = [0] * 11
                    for k in range(11):
                        action = self.a[k] 
                        action_true[k] = self.get_action(s1, s2, action)
                        print(s1, s2, action, action_true[k])
                        s1_start, s2_start = s1 + action_true[k], s2 - action_true[k]
                        #print(s1_start, s2_start)  
                        reward = np.sum(reward_transition_matrix_loc1[s1_start]) + np.sum(reward_transition_matrix_loc2[s2_start])
                        one_step_value = self.get_one_step_value(s1_start, s2_start, p_transition_matrix_loc1, p_transition_matrix_loc2)
                        q[i, j, k] = reward + one_step_value
                        print(reward, one_step_value)
                    # Improve the policy by acting greedily.
                    self.p[i, j] = action_true[np.argmax(q[i, j])]
                    print("greedy: ", q[i, j], np.argmax(q[i, j]), action_true[np.argmax(q[i, j])])
                    # Improve the value from any state over one greedy step.
                    self.v[i, j] = np.max(q[i, j])

        return self.p, self.v 


def draw_plots(p):
    ax = sns.heatmap(p, linewidth=0.5, annot=True, vmin=-5, vmax=5).invert_yaxis()
    plt.xlabel("loc1")
    plt.ylabel("loc2")
    plt.show()


def main():
    e = env()
#     p_transition_matrix, reward_transition_matrix = e.calculate_transition_matrix("loc1")
#     print(np.sum(p_transition_matrix, axis=0))
    #print(reward_transition_matrix[3], np.sum(reward_transition_matrix[3]))
    p, v = e.iteration(1)
    p = p*-1
    print(p)
#     print(v)
    draw_plots(p)


if __name__ == "__main__":
    main()
