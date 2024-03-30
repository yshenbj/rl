import numpy as np


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
        self.p = np.zeros((20, 20)) 
        self.v = np.zeros((20, 20))
        self.avg_req1, self.avg_ret1 = 3, 3
        self.avg_req2, self.avg_ret2 = 4, 2
        self.a = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    
    def get_reward(self, s1, s2):
        # Generate numbers for request and return among two places.
        n_req = np.random.poisson(self.avg_req1 + self.avg_req2)
        n_ret = np.random.poisson(self.avg_ret1 + self.avg_ret2)
        # Seprate into two parts (places).
        c_req = np.random.rand(n_req)
        c_ret = np.random.rand(n_ret)
        n_req1 = np.sum(c_req > self.avg_req1 / (self.avg_req1 + self.avg_req2))
        n_req2 = n_req - n_req1 
        n_ret1 = np.sum(c_ret > self.avg_ret1 / (self.avg_ret1 + self.avg_ret2))
        n_ret2 = n_ret - n_ret1 
        n_ren1, n_ren2 = np.minimum(n_req1, s1), np.minimum(n_req2, s2)
        # Get reward and next state.
        reward = 10 * (n_ren1 + n_ren2)
        s1, s2 = np.minimum(s1 - n_ren1 + n_ret1, 20), np.minimum(s2 - n_ren2 + n_ret2, 20)
        return reward, s1, s2
    
    def iteration(self, n_iter):
        for n in range(n_iter):
            q =  np.empty((20, 20, 11))
            for i in range(20):
                for j in range(20):
                    s1, s2 = i + 1, j + 1
                    # Act by policy
                    for k in range(11):
                        action = self.a[k] 
                        action = np.min([s1, s2, np.abs(action)]) * np.sign(action)
                        s1, s2 = np.minimum(s1 + action, 20), np.minimum(s2 - action, 20)
                        reward, s1, s2 = self.get_reward(s1, s2)
                        q[i, j, k] = reward + self.v[s1 - 1, s2 - 1]
                    # Improve the policy by acting greedily.
                    self.p[i, j] = self.a[np.argmax(q[i, j])]
                    # Improve the value from any state over one greddy step.
                    self.v[i, j] = np.max(q[i, j]) / (n + 1)
        return self.p, self.v 


def main():
    e = env()
    p, v = e.iteration(100)
    print(p)
    print(v)


if __name__ == "__main__":
    main()