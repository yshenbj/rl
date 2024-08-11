import numpy as np


class env():
    """
    Shortest Path in leacturer 3.
    Iterative application of Bellman optimality backup.
    """
    def __init__(self):
        self.goal = (0, 0)
        self.v = np.zeros((4, 4))
        self.p = 1
        self.r = -1
        self.gamma = 1

    def lookahead(self, i, j, a):
        if a == 0: 
            if j == 0: i1, j1 = i, j
            else: i1, j1 = i, j - 1
        elif a == 1:
            if i == 0: i1, j1 = i, j
            else: i1, j1 = i - 1, j
        elif a == 2:
            if j == 3: i1, j1 = i, j
            else: i1, j1 = i, j + 1 
        elif a == 3:
            if i == 3: i1, j1 = i, j
            else: i1, j1 = i + 1, j 
        return self.r + self.gamma * self.p * self.v[i1, j1]
    
    def iteration(self, n_iter):
        for _ in range(n_iter):
            for i in range(4):
                for j in range(4):
                    if (i == 0) and (j == 0):
                        continue 
                    else:
                        v_ls = []
                        for a in range(4):
                            v_ls.append(self.lookahead(i, j, a))
                        self.v[i, j] = np.max(v_ls) 
        return self.v 


def main():
    e = env()
    v = e.iteration(10)
    print(v)


if __name__ == "__main__":
    main()