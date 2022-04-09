import numpy as np
import matplotlib.pyplot as plt

y = np.loadtxt("err.txt")
x = np.loadtxt("h.txt")
fig, axs = plt.subplots()
axs.plot(x, y);
plt.yscale('log')
plt.grid(True)
plt.show()
