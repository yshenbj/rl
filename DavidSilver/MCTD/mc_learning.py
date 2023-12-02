import numpy as np 
import itertools
from tqdm import tqdm

class env():
    """
    Given a small gridword with size(m, n).
    Undiscounted episodic MDP (gamma=1).
    Agent follows uniform random policy.
    Rules are in the slides of leacture 3.
    """
    def __init__(self, reward=-1, size=(4, 4), terminals=[[0, 0], [3, 3]]):
        self.reward = reward
        self.size = size
        self.terminals = terminals
        self.values = np.zeros(size) 
        self.states_cnt = np.zeros(size)

    def action(self):
        # 0, 1, 2, 3 (w, n, e, s).
        return np.random.randint(4)

    def sampling(self):
        s_ls = []
        r_ls = []
        n = 0
        s = [np.random.randint(3), np.random.randint(3)]
        while s not in self.terminals:
            s_ls.append(s.copy())
            a = self.action()
            if a == 0:
                if s[1] > 0: s[1] -= 1
            elif a == 1:
                if s[0] > 0: s[0] -= 1
            elif a == 2:
                if s[1] < 3: s[1] += 1
            elif a == 3:
                if s[0] < 3: s[0] += 1
            r_ls.append(self.reward)
            n += 1
        g_ls = list(itertools.accumulate(r_ls))[::-1] 
        return s_ls, g_ls, n

    def evaluation(self, n_iter):
        for _ in tqdm(range(n_iter)):
            s_ls, g_ls, n = self.sampling() 
            for i in range(n):
                s = tuple(s_ls[i])
                self.states_cnt[s] += 1 
                self.values[s] += (g_ls[i] - self.values[s]) / self.states_cnt[s] 
        return self.values 
    

def main():
    e = env()
    v = e.evaluation(500000)
    print(v)


if __name__ == "__main__":
    main()
             
            
