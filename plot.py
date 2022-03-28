import numpy as np
import matplotlib.pyplot as plt


with open("Rewards.csv") as reward:
    array = np.loadtxt(reward, delimiter=",")

plt.plot(range(len(array)),array)
plt.show()