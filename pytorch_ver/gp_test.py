from matplotlib import pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

length_scale=1
noise=.1
kernel = RBF(length_scale=length_scale)+WhiteKernel(noise_level=noise**2)
gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)

x_max = 2
x_min = -2
n_observation=51


xs = np.zeros((10, n_observation))

xs[0] = (x_max-x_min) * (np.random.rand(n_observation) - .5)
idx = np.argsort(xs[0])
y = gp.sample_y(xs[0]).reshape(-1, 1)


plt.figure()
plt.plot(x[idx], y[idx])
plt.show()
