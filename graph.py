import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib.ticker as ticker

y1 = np.loadtxt("err_hamming_ofm.txt")
x1 = np.loadtxt("h_hamming_ofm.txt")
y2 = np.loadtxt("my_err_hamming_ofm.txt")
x2 = np.loadtxt("my_h_hamming_ofm.txt")
print(x1)
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
ax.plot(x1, y1)
ax.plot(x2, y2)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.005))
plt.legend(('Жесткое', 'Мягкое'))
ax.grid() 
plt.yscale("log")
plt.show()

