import numpy as np
import random

a = np.array([[2,3,6,2,2,6]])
print(random.choice([i for i in a if a[i] == 2]))

