import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib.ticker as ticker

x = np.array([0, 1, 2, 3])
y1 = np.loadtxt("errs_vector.txt")
y2 = np.loadtxt("errs_vector_my.txt")
print(y1, y2)
#fig, axs = plt.subplots()
#axs.plot(x1, y1, label='Жесткое')
#axs.plot(x2, y2, label='Мягкое')
#axs.legend()
#plt.yscale('log')
#plt.grid(True)
#plt.show()


fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlabel(r"$h^2$", fontsize=14)
ax.set_ylabel(r"$P(h^2)$", fontsize=14)
ax.plot(x, y1)
ax.plot(x, y2)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
plt.legend(('Жесткое', 'Мягкое'))
ax.grid() 
plt.show()

