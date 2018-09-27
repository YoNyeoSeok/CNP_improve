from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

length_scale=1
noise=.1
kernel = RBF(length_scale=length_scale)+WhiteKernel(noise_level=noise**2)
gp = GaussianProcessRegressor(kernel=kernel)#, optimizer=None)

x_max = 2
x_min = -2
n_observation=51


xs = np.zeros((10, n_observation))
window = np.linspace(-2, 2, 51)

xs[0] = (x_max-x_min) * (np.random.rand(n_observation) - .5)
idx = np.argsort(xs[0])
ys = gp.sample_y(xs[0].reshape(-1, 1))



#plt.figure()
#plt.scatter(xs[0][idx], ys[idx])
#plt.plot(xs[0][idx], ys[idx])
#plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

l = np.linspace(1, 1000, 11)
x1, x2 = np.meshgrid(l, l)
x = np.concatenate((x1.reshape(-1,1), x2.reshape(-1, 1)), axis=1)
y = gp.sample_y(x)

idx = np.random.permutation(11)
#x = x[idx]
#y = y[idx]
#gp.fit(x[idx],y[idx])
gp.fit(x,y)
print(x.shape, y.shape)

ax.plot_trisurf(x[:,0], x[:,1], gp.predict(x)[:,0])
ax.scatter(x[:,0], x[:,1], y[:,0])
plt.show()
