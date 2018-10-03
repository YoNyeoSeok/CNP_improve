from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class DataGenerator():
    def __init__(self, datasource, batch_size, \
            random_sample, disjoint_data, num_samples, num_samples_range, \
            input_range, window_range, window_step_size, random_window_position,
            task_limit, **kwags):
        self.datasource = datasource
        self.batch_size = batch_size
        self.random_sample = random_sample
        self.disjoint_data = disjoint_data 
        self.num_samples = num_samples
        self.num_samples_range = num_samples_range
        self.input_range = input_range
        self.window_range = window_range if window_range is not None else input_range
        self.window_step_size = window_step_size
        self.random_window_positino = random_window_position
        self.task_limit = task_limit
 
        if self.disjoint_data:
            self.gen_num_samples = sum(self.num_samples) if not self.random_sample else self.num_samples_range[1]*2
        else:
            self.gen_num_samples = max(self.num_samples) if not self.random_sample else self.num_samples_range[1]

        noise = .1
        length_scale = 1 
        kernel = RBF(length_scale=length_scale)+WhiteKernel(noise_level=noise**2)
        self.gp = gp = GaussianProcessRegressor(kernel=kernel)#, optimizer=None)

        if 'gp' in datasource:
            if '1d1d' in datasource:
                self.io_dims = [1, 1]
            elif '2d1d' in datasource:
                self.io_dims = [2, 1]

            if len(np.array(self.input_range).shape) == 1:
                self.input_range = np.repeat([self.input_range], [self.io_dims[0]], axis=0)
            if len(np.array(self.window_range).shape) == 1:
                self.window_range = np.repeat([self.window_range], [self.io_dims[0]], axis=0)

            self.fs = [gp.sample_y]
            for i in range(args.task_limit):
                x = np.random.rand(self.gen_num_samples, self.io_dims[0])
                x = (self.input_range[:,1] - self.input_range[:,0])*x + self.input_range[:,0]

                y = gp.sample_y(x)
                f = gp.fit(x, y)
                self.fs += [lambda x: gp.predict(x) + noise*np.random.randn((*x.shape[:-1], self.io_dims[1]))]


        if datasource == 'gp1d':
            self.io_dims = [1, 1]
            #y_mu, y_cov = gp.predict(x_plot, return_cov=True)

            if task_limit != 0:
                pass
                self.xs = xs = np.zeros((task_limit, self.gen_num_samples, self.io_dims[0]))
                self.ys = ys = np.zeros((task_limit, self.gen_num_samples, self.io_dims[1]))
                for i in range(self.task_limit):
                    self.xs[i] = xs[i] = (input_range[1]-input_range[0]) * (np.random.rand(num_samples, self.io_dims[0]) - .5)
                    self.ys[i] = ys[i] = gp.sample_y(xs[i])

#        elif datasource == 'sinusoidal':
        elif datasource == 'gp1d1d':
            self.io_dims = [1, 1]
            self.f = lambda x: self.gp.sample_y(x.reshape(-1, 1)).reshape(x.shape) + noise*np.random.randn(*x.shape)
        elif datasource == 'gp2d1d':
            self.io_dims = [2, 1]
            if len(np.array(self.input_range).shape) == 1:
                self.input_range = np.repeat([self.input_range], [2], axis=0)
            if len(np.array(self.window_range).shape) == 1:
                self.window_range = np.repeat([self.window_range], [2], axis=0)
            self.f = lambda x: self.gp.sample_y(x.reshape(-1, self.io_dims[0])).reshape((*x.shape[:-1], self.io_dims[1])) \
                    + noise*np.random.randn((*x.shape[:-1], self.io_dims[1]))
            #self.f = lambda x1, x2: self.gp.sample_y(np.concatenate(
            #    (x1.reshape(-1, 1), x2.reshape(-1, 1)), axis=1)).reshape(x1.shape) + noise*np.random.randn(*x1.shape)

        elif datasource == 'branin':
            self.io_dims = [2, 1]
            if len(np.array(self.input_range).shape) == 1:
                self.input_range = np.repeat([self.input_range], [2], axis=0)
            if len(np.array(self.window_range).shape) == 1:
                self.window_range = np.repeat([self.window_range], [2], axis=0)

            self.a = a = 1
            self.b = b = 5.1/(4*np.pi**2)
            self.c = c = 5/np.pi
            self.r = r = 6
            self.s = s = 10
            self.t = t = 1/(8*np.pi)
            self.f = f = lambda x1, x2: a*(x2-b*x1**2+c*x1-r)**2 + s*(1-t)*np.cos(x1) + s + noise*np.random.randn(*x1.shape)
            #            def fun(x):
            #                if len(x)==2:
            #                    x1,x2 = x
            #                elif len(x.T)==2:
            #                    x1,x2 = x
            #                return a*(x2-b*x1**2+c*x1-r)**2 + s*(1-t)*np.cos(x1) + s + noise*np.random.randn(*x1.shape)
            #            self.f = f = lambda x: a*(x2-b*x1**2+c*x1-r)**2 + s*(1-t)*np.cos(x1) + s + noise*np.random.randn(*x1.shape)
            #            self.f = lambda x: self.gp.sample_y(x.reshape(-1, self.io_dims[0]).reshape((*x.shape[:-1], self.io_dims[1])) \
                    #                    + noise*np.random.randn((*x.shape[:-1], self.io_dims[1]))

            if task_limit != 0:
                pass
                self.xs = xs = np.zeros((task_limit, self.gen_num_samples, self.io_dims[0]))
                self.ys = ys = np.zeros((task_limit, self.gen_num_samples, self.io_dims[1]))
                for i in range(self.task_limit):
                    self.xs[i] = xs[i] = (input_range[1]-input_range[0]) * (np.random.rand(num_samples, self.io_dims[0]) - .5)
                    self.ys[i] = ys[i] = f(xs[i][0], xs[i][1])

        if self.task_limit != 0 and 'gp' in datasource:
            self.fs = [self.f]
            for i in range(self.task_limit):
                x = (self.input_range[1]-self.input_range[0]) * np.random.rand(self.gen_num_samples, self.io_dims[0]) + self.input_range[0]
                y = self.gp.sample_y(x)
                f = self.gp.fit(x, y)
                self.fs += [lambda x: f.predict(x)]

    def get_task_idx(self, task_limit=None):
        if task_limit is None:
            task_limit = self.task_limit

        return np.random.randint(0 if task_limit is 0 else 1, task_limit+1)

    def get_train_test_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        xs, ys = self.generate_batch(batch_size)

        if self.disjoint_data:
            if self.random_sample:
                Ns_train = np.random.randint(self.num_samples_range[0], self.num_samples_range[1], batch_size)
                Ns_test = np.random.randint(self.num_samples_range[0], self.num_samples_range[1], batch_size)+self.num_samples_range[1]
                x_train_batch, x_test_batch = zip(*map(lambda x, N_train, N_test: 
                        (x[:N_train], x[self.num_samples_range[1]:N_test]), xs, Ns_train, Ns_test))
                y_train_batch, y_test_batch = zip(*map(lambda y, N_train, N_test: 
                        (y[:N_train], y[self.num_samples_range[1]:N_test]), ys, Ns_train, Ns_test))
            else:
                x_train_batch, x_test_batch = zip(*map(lambda x: 
                        (x[:self.num_samples[0]], x[self.num_samples[0]:]), xs))
                y_train_batch, y_test_batch = zip(*map(lambda y: 
                        (y[:self.num_samples[0]], y[self.num_samples[0]:]), ys))
        else:
            if self.random_sample:
                Ns = np.random.randint(self.num_samples_range[0], self.num_samples_range[1], batch_size)
                x_train_batch, x_test_batch = zip(*map(lambda x, N: (x[:N], x), xs, Ns))
                y_train_batch, y_test_batch = zip(*map(lambda y, N: (y[:N], y), ys, Ns))
            else:
                x_train_batch, x_test_batch = zip(*map(lambda x: 
                        (x[:self.num_samples[0]], x[:self.num_samples[1]]), xs))
                y_train_batch, y_test_batch = zip(*map(lambda y: 
                        (y[:self.num_samples[0]], y[:self.num_samples[1]]), ys))

#        print(len(x_train_batch[0]), len(y_train_batch[0]), len(x_test_batch[0]), len(y_test_batch[0]))
        return (x_train_batch, y_train_batch), (x_test_batch, y_test_batch)

    def get_train_test_sample(self, x_y=None):
        train_batch, test_batch = self.get_train_test_batch(batch_size=1)
        x_train_batch, y_train_batch = train_batch
        x_test_batch, y_test_batch = test_batch
        return [x_train_batch[0], y_train_batch[0]], [x_test_batch[0], y_test_batch[0]]

    def generate_batch(self, batch_size=None, num_samples=None):
        if batch_size is None:
            batch_size = self.batch_size
        if num_samples is None:
            num_samples = self.gen_num_samples

#        xs = np.zeros((batch_size, num_samples, self.io_dims[0]))
#        ys = np.zeros((batch_size, num_samples, self.io_dims[1]))

#        xs = (self.input_range[1]-self.input_range[0]) * np.random.rand(batch_size, num_samples, self.io_dims[0]) + self.input_range[0]
#        idx = self.get_task_idx()
#        ys = self.fs[idx](xs) + np.random.randn(batch_size, num_samples, self.io_dims[1])

        if self.task_limit != 0:
            if 'gp' in args.datasource:
                i = np.random.randint(0, self.task_limit, batch_size)
                if self.io_dims == [1, 1]:
                    xs = (self.input_range[1]-self.input_range[0]) * np.random.rand(batch_size, num_samples, self.io_dims[0]) + self.input_range[0]
            elif self.io_dims == [2, 1]:
                x = self.xs[i]
                y = self.ys[i]
        else:
            if self.io_dims == [1, 1]:
                xs = (self.input_range[1]-self.input_range[0]) * np.random.rand(batch_size, num_samples, self.io_dims[0]) + self.input_range[0]
                ys = map(self.f, xs)
                #ys = self.f(xs) + np.random.randn(batch_size, num_samples, self.io_dims[1])
                #ys = np.array([self.gp.sample_y(x) for x in xs])
            elif self.io_dims == [2, 1]:
                #elif self.datasource == 'branin':
                x1s = (self.input_range[0][1]-self.input_range[0][0]) * np.random.rand(batch_size, num_samples, 1) + self.input_range[0][0]
                x2s = (self.input_range[1][1]-self.input_range[1][0]) * np.random.rand(batch_size, num_samples, 1) + self.input_range[1][0]
                #ys = self.f(x1s, x2s) + np.random.randn(batch_size, num_samples, self.io_dims[1])
                ys = map(self.f, x1s, x2s)
                xs = np.concatenate((x1s, x2s), axis=2)
#        print(len(list(xs)))
#        print(len(list(ys)))
        return xs, ys

    def generate_sample(self, num_samples=None):
        xs, ys = self.generate_batch(1)
        return xs[0], ys[0]
    
    def generate_window_samples(self, window_range=None, step_size=None):
        if window_range is None:
            window_range = self.input_range
        if step_size is None:
            step_size = self.window_step_size

        if self.io_dims[0] == 1:
            r = window_range[1] - window_range[0]
            return np.linspace(window_range[0], window_range[1], int(r/step_size + 1)).reshape(-1, 1)
            #return np.linspace(window_range[0], window_range[1], num_samples).reshape(-1, 1)
        elif self.io_dims[0] == 2:
            r1 = window_range[0][1] - window_range[0][0]
            l1 = np.linspace(window_range[0][0], window_range[0][1], int(r1/step_size + 1)).reshape(-1, 1)
            r2 = window_range[1][1] - window_range[1][0]
            l2 = np.linspace(window_range[1][0], window_range[1][1], int(r2/step_size + 1)).reshape(-1, 1)
            x1, x2 = np.meshgrid(l1, l2)
            return np.concatenate((x1.reshape(-1, 1), x2.reshape(-1, 1)), axis=1)

    def make_fig_title(self, fig, title="test"):
        fig.suptitle(title)

    def make_fig_ax(self, fig):
        if sum(self.io_dims) == 2:
            ax = fig.add_subplot(111)
            ax.set_xlim(self.window_range[0], self.window_range[1])
            #ax.set_xlim(self.input_range[0], self.input_range[1])
            #ax.set_ylim(self.output_range[0], self.output_range[1])
        elif sum(self.io_dims) == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim(self.window_range[0][0], self.window_range[0][1])
            ax.set_ylim(self.window_range[1][0], self.window_range[1][1])
            #ax.set_xlim(self.input_range[0], self.input_range[1])
            #ax.set_ylim(self.input_range[0], self.input_range[1])
            #ax.set_zlim(self.output_range[0], self.output_range[1])
        return ax

    def window_crop(self, x, window_range=None):
        if window_range is None:
            window_range = self.window_range
        if len(np.array(window_range).shape) == 1:
            return np.logical_and(np.ones(len(x)), *[np.logical_and(window_range[0]<=t, t<=window_range[1]) \
                for i, t in enumerate(x.T)])
        return np.logical_and(*[np.logical_and(window_range[i][0]<=t, t<=window_range[i][1]) \
            for i, t in enumerate(x.T)])

    def plot_data(self, ax, data, c1='k', c2='gray', window_crop=False):
        if data.shape[1] == 3:
            if window_crop:
                data = data[self.window_crop(data[:,:1])]
            ax.plot(data[:,0], data[:,1], color=c1)
            ax.fill_between(data[:,0], 
                    data[:,1] - np.sqrt(data[:,2]), 
                    data[:,1] + np.sqrt(data[:,2]),
                    alpha=.5, color=c2)
        elif data.shape[1] == 4:
            data = data[self.window_crop(data[:,:2])]
            ax.plot_trisurf(data[:,0], data[:,1], data[:,2], color=c2)
        pass
    def contour_data(self, ax, data, c1='k', c2='gray', window_crop=False):
        if data.shape[1] == 3:
            if window_crop:
                data = data[self.window_crop(data[:,:1])]
            ax.plot(data[:,0], data[:,1], color=c1)
            ax.fill_between(data[:,0], 
                    data[:,1] - np.sqrt(data[:,2]), 
                    data[:,1] + np.sqrt(data[:,2]),
                    alpha=.5, color=c2)
        elif data.shape[1] == 4:
            data = data[self.window_crop(data[:,:2])]
            N_ = len(data)
            n = int(np.sqrt(N_))
            CS = ax.contour(data[:,0].reshape(n, n), data[:,1].reshape(n, n), data[:,2].reshape(n,n), color=c2)
            ax.clabel(CS, inline=1, fontsize=10)

    def plot_gp(self, ax, train, data, c='c'):
        if self.io_dims == [1, 1]:
            train = train[self.window_crop(train[:,:1])]
            data = data[self.window_crop(data[:,:1])]
            self.gp.fit(train[:,:1], train[:,1:])
            ax.plot(data[:,:1], self.gp.predict(data[:,:1]), color=c)
        elif self.io_dims == [2, 1]:
            #print('shape', train.shape, data.shape)
            train = train[self.window_crop(train[:,:2])]
            data = data[self.window_crop(data[:,:2])]
            #print('shape2', train.shape, data.shape)
            #print('data', train, data)
            #train[:,2:] = self.gp.sample_y(train[:,:2])
            self.gp.fit(train[:,:2], train[:,2:])       ### need sort
            #print(train[:,:2].shape, train[:,2:].shape)
            #print(data)
            #print(data[:,:1])
            #print(data[:,1:2])
            #print(self.gp.predict(data[:,:2]))
            #ax.plot_trisurf(train[:,0], train[:,1], self.gp.predict(train[:,:2])[:,0], color='k')
            #ax.plot_trisurf(train[:,0], train[:,1], self.f(train[:,:1], train[:,1:2]).reshape(-1,), color='k')
            ax.plot_trisurf(data[:,0], data[:,1], self.gp.predict(data[:,:2])[:,0], color=c)

    def plot_branin(self, ax, data, c='c'):
        if self.datasource == 'branin':
            data = data[self.window_crop(data[:,:1])]
            ax.plot_trisurf(data[:,0], data[:,1], self.f(data[:,:1], data[:,1:2]), color=c)

    def scatter_data(self, ax, data, c='k'):
        if data.shape[1] == 2:
            data = data[self.window_crop(data[:,:1])]
            ax.scatter(data[:,0], data[:,1], c=c)
        elif data.shape[1] == 3:
            data = data[self.window_crop(data[:,:2])]
            ax.scatter(data[:,0], data[:,1], data[:,2], c=c) 
        pass

