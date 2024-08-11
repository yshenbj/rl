import numpy as np


class env1():
    """
    Given a small gridword with size(m, n), do policy evaluation based on 
    Bellman expectation equation. 
    Undiscounted episodic MDP (gamma=1).
    Rules are in the slides of leacture 3.
    Iteration form.
    """
    def __init__(self, reward=-1, size=(4, 4), terminals=[(0, 0), (3, 3)]):
        self.reward = reward
        self.size = size
        self.terminals = terminals
        self.values = np.zeros(size)

    def get_value(self, loc, val):
        if (loc[0] not in range(self.size[0])) | (loc[1] not in range(self.size[1])):
            return val + self.reward
        elif loc in self.terminals:
            return self.reward
        else:
            return self.values[loc] + self.reward

    def policy_value(self, loc):
        if loc in self.terminals:
            return 0
        val = self.values[loc] 
        next_locs = [(loc[0]+1, loc[1]), (loc[0]-1, loc[1]), (loc[0], loc[1]+1), (loc[0], loc[1]-1)]
        next_values = []
        for next_loc in next_locs:
            next_values.append(self.get_value(next_loc, val))
        return np.mean(next_values)
        

    def evaluation(self, n_iter):
        for _ in range(n_iter):
            update_values = np.empty(self.size)
            for i in range(self.size[0]):
                for j in range(self.size[1]): 
                    update_values[i, j] = self.policy_value((i, j))
            self.values = update_values 
        return self.values
    

class env2():
    """
    Given a small gridword with size(m, n), do policy evaluation based on 
    Bellman expectation equation. 
    Undiscounted episodic MDP ()
    Rules are in the slides of leacture 3.
    Matrix form.
    """    
    def __init__(self, reward=-1, size=(4, 4), terminals=[(0, 0), (3, 3)]):
        self.size = size
        self.V = np.zeros(np.prod(size)) 
        self.R = np.ones(size) * reward
        for t in terminals: 
            self.R[t] = 0 
        self.R = self.R.reshape(-1)
        self.P = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                [0.25, 0.25, 0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                [0, 0.25, 0.25, 0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0.25, 0.5, 0, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0], 
                [0.25, 0, 0, 0, 0.25, 0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0, 0], 
                [0, 0.25, 0, 0, 0.25, 0, 0.25, 0, 0, 0.25, 0, 0, 0, 0, 0, 0], 
                [0, 0, 0.25, 0, 0, 0.25, 0, 0.25, 0, 0, 0.25, 0, 0, 0, 0, 0],
                [0, 0, 0, 0.25, 0, 0, 0.25, 0.25, 0, 0, 0, 0.25, 0, 0, 0, 0], 
                [0, 0, 0, 0, 0.25, 0, 0, 0, 0.25, 0.25, 0, 0, 0.25, 0, 0, 0], 
                [0, 0, 0, 0, 0, 0.25, 0, 0, 0.25, 0, 0.25, 0, 0, 0.25, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0.25, 0, 0, 0.25, 0, 0.25, 0, 0, 0.25, 0], 
                [0, 0, 0, 0, 0, 0, 0, 0.25, 0, 0, 0.25, 0.25, 0, 0, 0, 0.25], 
                [0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0, 0, 0, 0.5, 0.25, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0, 0, 0.25, 0.25, 0.25, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0, 0, 0.25, 0.25, 0.25],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]
        )

    def update_all(self):
        self.V = self.R + 1 * np.dot(self.P, self.V)     

    def evaluation(self, n_iter):
        for _ in range(n_iter):
            self.update_all()
        return self.V.reshape(self.size)
    
    def solve(self):
        return np.dot(np.linalg.inv(np.identity(self.P.shape[0]) - 1 * self.P), self.R).reshape(self.size)


def main():
    e1 = env1()
    values = e1.evaluation(1000)
    print(values)

    e2 = env2()
    values = e2.evaluation(1000)
    print(values)

    values = e2.solve()
    print(values)


if __name__ == "__main__":
    main()