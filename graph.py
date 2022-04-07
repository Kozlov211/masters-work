import numpy as np
import matplotlib.pyplot as plt

y1 = np.loadtxt("norm_0.txt")
y2 = np.loadtxt("norm_1.txt")
x = np.loadtxt("abscissa.txt")
fig, axs = plt.subplots()
axs.plot(x, y1)
axs.plot(x, y2, '--')
plt.grid(True)
plt.show()
