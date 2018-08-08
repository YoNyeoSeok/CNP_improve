import numpy as np
from matplotlib import pyplot as plt

class Figure():
    def __init__(self, show=False):
        self.show = show
        self.fig = plt.figure()

    def set_axis(self, xlim=[-2,2], ylim=[-3,3]):
        self.xlim = xlim
        self.ylim = ylim
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])

    def plot_xy(self, x, y, color='k'):
        plt.plot(x, y, color='k')

    def scatter_xy(self, x, y, color='k'):
        plt.scatter(x, y, c=color)

    def plot_fill_xy(self, x, y_mu, y_cov, color_l='k', color_f='k',
            alpha=.5):
        plt.plot(x, y_mu, color=color_l)
        plt.fill_between(x,
                y_mu - np.sqrt(np.diag(y_cov)),
                y_mu + np.sqrt(np.diag(y_cov)),
                alpha=alpha, color=color_f)

