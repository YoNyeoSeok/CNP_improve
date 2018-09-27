from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import numpy as np

gp = GaussianProcessRegressor()
x = np.linspace(-2, 2, 4*5)[:, None]
y = gp.sample_y(x, random_state = 1)

plt.figure()
plt.plot(x, y)
plt.show()
