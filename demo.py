import numpy as np
import random
import itertools
# a = np.arange(12).reshape(2, 2, 3) + 10
# print(np.argmax(a, axis=2)) 

# a = ([2, 3, 4, 5, 6, 7, 8, 9] * 4 + [10] * 12 + ['a'] * 4) * 2
# random.shuffle(a) 
# x = a.pop()

# x = (np.random.randint(3), np.random.randint(3))
# print(x)

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a[tuple([1,1])])