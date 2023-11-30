import numpy as np
import random
# a = np.arange(12).reshape(2, 2, 3) + 10
# print(np.argmax(a, axis=2)) 

a = [2, 3, 4, 5, 6, 7, 8, 9] * 4 + [10] * 12 + ['a'] * 4
random.shuffle(a)
print(a)