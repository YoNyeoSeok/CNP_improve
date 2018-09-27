import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

x_min = 0
x_max = 1

X = np.linspace(x_min, x_max, 1)
X_ = X[:, None]
y = np.random.rand(X_.shape[0])
y = np.sin(X*2*np.pi) + np.random.rand(X.shape[0])*.1
y_ = y[:, None] 

X_test = np.linspace(x_min, x_max, 51)
X_test_ = X_test[:, None]

sigma = .1
sigma_f = 1
length_scale = .1
kernel = sigma_f * RBF(length_scale=length_scale) + WhiteKernel(noise_level=sigma**2)
gp = GaussianProcessRegressor(kernel=kernel).fit(X_, y_)

y_mean, y_cov = gp.predict(X_test_, return_cov=True)
print(gp.log_marginal_likelihood(gp.kernel_.theta), gp.kernel_.theta)
print(np.diag(y_cov))

plt.figure()
plt.plot(X_test, y_mean, 'k', lw=3, zorder=9)
plt.fill_between(X_test_.reshape(-1,), y_mean.reshape(-1,) - np.sqrt(np.diag(y_cov)),
                 y_mean.reshape(-1,) + np.sqrt(np.diag(y_cov)),
                 alpha=0.5, color='k')
#plt.plot(X_test, f(X_test), 'r', lw=3, zorder=9)
plt.scatter(X, y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
          % (kernel, gp.kernel_,
             gp.log_marginal_likelihood(gp.kernel_.theta)))
plt.tight_layout()
plt.show()

