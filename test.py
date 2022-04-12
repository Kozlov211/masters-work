import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib.ticker as ticker

y1 = np.loadtxt("signal.txt")

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
ax.plot(y1)
plt.show()

