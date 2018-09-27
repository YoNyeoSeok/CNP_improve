from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class DataGenerator():
    def __init__(self, datasource, batch_size, \
            random_sample, disjoint_data, num_samples, num_samples_range, \
            input_range, output_range, task_limit):
        self.datasource = datasource
        self.batch_size = batch_size
        self.random_sample = random_sample
        self.disjoint_data = disjoint_data 
        self.num_samples = num_samples
        self.num_samples_range = num_samples_range
        self.input_range = input_range
        self.output_range = output_range
        self.task_limit = task_limit
 
        if self.disjoint_data:
            self.gen_num_samples = sum(self.num_samples) if not self.random_sample else self.num_samples_range[1]*2
        else:
            self.gen_num_samples = max(self.num_samples) if not self.random_sample else self.num_samples_range[1]

        if datasource == 'gp1d':
            self.io_dims = [1, 1]

            length_scale = 1 
            noise = .1
            kernel = RBF(length_scale=length_scale)+WhiteKernel(noise_level=noise**2)
            self.gp = gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)

            #y_mu, y_cov = gp.predict(x_plot, return_cov=True)

            if task_limit != 0:
                pass
                self.xs = xs = np.zeros((task_limit, self.gen_num_samples, self.io_dims[0]))
                self.ys = ys = np.zeros((task_limit, self.gen_num_samples, self.io_dims[1]))
                for i in range(self.task_limit):
                    self.xs[i] = xs[i] = (input_range[1]-input_range[0]) * (np.random.rand(num_samples, self.io_dims[0]) - .5)
                    self.ys[i] = ys[i] = gp.sample_y(xs[i])

        elif datasource == 'branin':
            self.io_dims = [2, 1]

            self.a = a = 1
            self.b = b = 5.1/(4*np.pi**2)
            self.c = c = 5/np.pi
            self.r = r = 6
            self.s = s = 10
            self.t = t = 1/(8*np.pi)
            self.f = f = lambda x, y: a*(y-b*x**2+c*x-r) + s*(1-t)*np.cos(x) + s #\
                    #                                   if x.ndim > 2 else \
                    #                                   a*(x[:,1,None]-b*x[:,0,None]**2+c*x[:,0,None]-r) + s*(1-t)*np.cos(x[:,0,None]) + s 

            if task_limit != 0:
                pass
                self.xs = xs = np.zeros((task_limit, self.gen_num_samples, self.io_dims[0]))
                self.ys = ys = np.zeros((task_limit, self.gen_num_samples, self.io_dims[1]))
                for i in range(self.task_limit):
                    self.xs[i] = xs[i] = (input_range[1]-input_range[0]) * (np.random.rand(num_samples, self.io_dims[0]) - .5)
                    self.ys[i] = ys[i] = f(xs[i][0], xs[i][1])
    def get_train_test_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        trains = [np.zeros(batch_size, self.gen_num_samples, self.io_dims[0]), \
                np.zeros(batch_size, self.gen_num_samples, self.io_dims[1])]
        tests = [np.zeros(batch_size, self.gen_num_samples, self.io_dims[0]), \
                np.zeros(batch_size, self.gen_num_samples, self.io_dims[1])]
        for i in range(batch_size):
            trains[i], tests[i] = self.get_train_test_sample()
        return trains, tests

    def get_train_test_sample(self, x_y=None):
        if x_y is None:
            x, y = self.generate_sample()
        else:
            x, y = x_y

        if self.disjoint_data:
            if self.random_sample:
                N = np.random.randint(self.num_samples_range[0], self.num_samples_range[1])
                x_train = x[:N]
                y_train = y[:N]
                N = self.num_samples_range[1] + np.random.randint(self.num_samples_range[0], self.num_samples_range[1])
                x_test = x[self.num_samples_range[1]:N]
                y_test = y[self.num_samples_range[1]:N]
            else:
                x_train = x[:self.num_samples[0]]
                y_train = y[:self.num_samples[0]]
                x_test = x[self.num_samples[0]:]
                y_test = y[self.num_samples[0]:]
        else:
            if self.random_sample:
                N = np.random.randint(self.num_samples_range[0], self.num_samples_range[1])
                x_train = x[:N]
                y_train = y[:N]
                x_test = x
                y_test = y
            else:
                x_train = x[:self.num_samples[0]]
                y_train = y[:self.num_samples[0]]
                x_test = x[:self.num_samples[1]]
                y_test = y[:self.num_samples[1]]

        return [x_train, y_train], [x_test, y_test]

    def generate_sample(self, num_samples=None):
        if num_samples is None:
            num_samples = self.gen_num_samples
        if self.task_limit != 0:
            pass
            i = np.random.randint(self.task_limit)
            x = self.xs[i]
            y = self.ys[i]
        else:
            if self.datasource == 'gp1d':
                x = (self.input_range[1]-self.input_range[0]) * np.random.rand(num_samples, self.io_dims[0]) + self.input_range[0]
                y = self.gp.sample_y(x).reshape(-1, 1)
            elif self.datasource == 'branin':
                x = (self.input_range[1]-self.input_range[0]) * np.random.rand(num_samples, self.io_dims[0]) + self.input_range[0]
                y = self.f(x[:,None,0], x[:,None,1])

        return x, y
    
    def generate_space_sample(self, num_samples=None, step_size=None):
        if num_samples is None:
            num_samples = self.gen_num_samples
        if step_size is None:
            step_size = 1
        if self.io_dims[0] == 1:
            #return np.linspace(self.input_range[0], self.input_range[1], num_samples).reshape(-1, 1)
            r = self.input_range[1] - self.input_range[0]
            return np.linspace(self.input_range[0], self.input_range[1], int(r/step_size + 1)).reshape(-1, 1)
        elif self.io_dims[0] == 2:
            r = self.input_range[1] - self.input_range[0]
            l = np.linspace(self.input_range[0], self.input_range[1], int(r/step_size + 1)).reshape(-1, 1)
            #l = np.linspace(self.input_range[0], self.input_range[1], int(np.sqrt(num_samples)))
            x1, x2 = np.meshgrid(l, l)
            return np.concatenate((x1.reshape(-1, 1), x2.reshape(-1, 1)), axis=1)
            return np.array(zip(*(x.flat for x in np.meshgrid(l, l))))
            return np.vstack([x1.ravel(), x2.ravel()])
            return np.concatenate(np.meshgrid(l, l), axis=0).reshape(-1, 2)



    def generate_batch(self, batch_size=None, num_samples=None):
        if batch_size is None:
            batch_size = self.batch_size
        if num_samples is None:
            num_samples = self.num_samples

        xs = np.zeros((batch_size, num_samples, self.io_dims[0]))
        ys = np.zeros((batch_size, num_samples, self.io_dims[1]))
        for i in range(batch_size):
            xs[i], ys[i] = self.generate_sample(num_samples)

        return xs, ys

    def make_fig_ax(self, fig):
        if sum(self.io_dims) == 2:
            ax = fig.add_subplot(111)
            ax.set_xlim(self.input_range[0], self.input_range[1])
            ax.set_ylim(self.output_range[0], self.output_range[1])
        elif sum(self.io_dims) == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim(self.input_range[0], self.input_range[1])
            ax.set_ylim(self.input_range[0], self.input_range[1])
            #ax.set_zlim(self.output_range[0], self.output_range[1])
        return ax
            

    def plot_data(self, ax, data, c1='k', c2='gray'):
        if data.shape[1] == 3:
            ax.plot(data[:,0], data[:,1], color=c1)
            ax.fill_between(data[:,0], 
                    data[:,1] - np.sqrt(data[:,2]), 
                    data[:,1] + np.sqrt(data[:,2]),
                    alpha=.5, color=c2)
        elif data.shape[1] == 4:
            ax.plot_trisurf(data[:,0], data[:,1], data[:,2], color=c2)
        pass
    def scatter_data(self, ax, data, c='k'):
        if data.shape[1] == 2:
            ax.scatter(data[:,0], data[:,1], c=c)
        elif data.shape[1] == 3:
            ax.scatter(data[:,0], data[:,1], data[:,2], c=c) 
        pass 

    
