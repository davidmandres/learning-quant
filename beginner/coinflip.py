import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

coinflip = np.random.choice([-1, 1], size=(100000,))

plt.style.use("seaborn-v0_8-dark")
mpl.rcParams['font.family'] = 'serif'

plt.plot(coinflip.cumsum(), label="Random Walk")
plt.title("Coin Flip Cumulative Sum (Random Walk)")
plt.xlabel("Number of Flips")
plt.ylabel("Position")
plt.legend()
plt.show()


