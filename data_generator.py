from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class Dist():
    def __init__(self, io_dims, input_range):
        self.io_dims = io_dims
        self.input_range = input_range
        self.gen_num_samples = gen_num_samples
        pass
    def gen_param(self, seed):
        pass
    def f_(self, x, param):
        pass
    def f(self, x):
        pass

class GP():
    def __init__(self, io_dims, input_range, gen_num_samples):
        self.io_dims = io_dims
        self.input_range = input_range
        self.gen_num_samples = gen_num_samples

        noise = .1
        length_scale = 1 
        kernel = RBF(length_scale=length_scale)+WhiteKernel(noise_level=noise**2)
        self.gp = lambda : GaussianProcessRegressor(kernel=kernel)#, optimizer=None)

        self.param = self.gen_param()

    def gen_param(self, seed=0):
        x = np.random.rand(self.gen_num_samples, self.io_dims[0])
        x = (self.input_range[:,1] - self.input_range[:,0])*x + self.input_range[:,0]
        y = self.gp().sample_y(x)
        return x, y

    def f_(self, x, param):
        return self.gp().fit(*param).predict(x)

    def f(self, x):
        return self.f_(x, self.param)
    
    def save_task(self, fname):
        self.param = np.save(self.param, fname)
    def load_task(self, fname):
        self.param = np.load(fname)

class BRANIN():
    def __init__(self, io_dims, input_range, gen_num_samples):
        self.io_dims = io_dims
        self.input_range = input_range
        self.gen_num_samples = gen_num_samples
        self.param = self.gen_param()

    def gen_param(self, seed=0):
        if seed == 0:
            a, b, c, r, s, t = 1, 5.1/(4*np.pi**2), 5/np.pi, 6, 10, 1/(8*np.pi)
        else:
            a, b, c, r, s, t = 1, 5.1/(4*np.pi**2), 5/np.pi, 6, 100*seed, 1/(8*np.pi)
        return (a, b, c, r, s, t)

    def f_(self, x, param):
        a, b, c, r, s, t = param
        x1, x2 = np.split(x, 2, 1)
        return a*(x2-b*x1**2+c*x1-r)**2 + s*(1-t)*np.cos(x1) + s

    def f(self, x):
        return self.f_(x, self.param)
         
    def save_task(self, fname):
        self.param = np.save(self.param, fname)
    def load_task(self, fname):
        self.param = np.load(fname)
#    self.f = f
#    a = 1
#    self.b = b = 5.1/(4*np.pi**2)
#    self.c = c = 5/np.pi
#    self.r = r = 6
#    self.s = s = 10
#    self.t = t = 1/(8*np.pi)
#   #self.f = f = lambda x1, x2: a*(x2-b*x1**2+c*x1-r)**2 + s*(1-t)*np.cos(x1) + s + noise*np.random.randn(*x1.shape)

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
        self.gp = lambda: GaussianProcessRegressor(kernel=kernel)#, optimizer=None)
        gp = self.gp()

        if 'gp' in datasource:
            if '1d1d' in datasource:
                self.io_dims = [1,1]
            elif '2d1d' in datasource:
                self.io_dims = [2,1]
        elif datasource == 'branin':
            self.io_dims = [2, 1]
            
        if len(np.array(self.input_range).shape) == 1:
            self.input_range = np.repeat([self.input_range], [self.io_dims[0]], axis=0)
        if len(np.array(self.window_range).shape) == 1:
            self.window_range = np.repeat([self.window_range], [self.io_dims[0]], axis=0)

        if 'gp' in datasource:
            self.task = GP(self.io_dims, self.input_range, self.gen_num_samples)
        elif datasource == 'branin':
            self.task = BRANIN(self.io_dims, self.input_range, self.gen_num_samples)

        input_shape = (-1, self.io_dims[0])
        output_shape = lambda x: (*x.shape[:-1], self.io_dims[1])

        self.fb = lambda f, x: f(x.reshape(*input_shape)).reshape(*output_shape(x))
        def fn(f, x):
            y = f(x)
            return y + noise*np.random.randn(*y.shape)
        self.fn = fn

        self.fbn = lambda f, xs: self.fb((lambda x: self.fn(f, x)), xs)

        # task
        # task 0 generate random
        self.params = [[]]*(self.task_limit+1)
        def fs(xs):
            self.params[0] = self.task.gen_param(seed=np.random.randint(1000))
            return self.fbn((lambda x: self.task.f_(x, self.params[0])), xs)
        self.fs = [fs]

        for i in range(1, self.task_limit+1):
            self.params[i] = (self.task.gen_param(seed=i))
            self.fs += [lambda xs: self.fbn((lambda x: self.task.f_(x, self.params[i])), xs)]

        self.fs = np.array(self.fs)
        """ 
        return
        self.params = [(task.gen_param())]
        def fs(x):
            self.params[0] = (self.gen_param())
            return self.fn(x.reshape(*input_shape), self.params[0]).reshape(*output_shape(x))
        self.fs = [fs]
        #self.fs = [lambda x: self.fn(x.reshape(*input_shape), self.gen_param()).reshape(*output_shape(x))]

        for i in range(self.task_limit):
            self.params += [self.gen_param()]
            self.fs += [lambda x: self.fn(x.reshape(*input_shape), self.params[i+1]).reshape(*output_shape(x))]
        self.fs = np.array(self.fs)

        elif datasource == 'branin':
            def gen_param():
                try:
                    gen_param.first_call
                    a, b, c, r, s, t = 1, 5.1/(4*np.pi**2), 5/np.pi, 6, 10, 1/(8*np.pi)
                except AttributeError:
                    gen_param.first_call = True
                    a, b, c, r, s, t = 1, 5.1/(4*np.pi**2), 5/np.pi, 6, 100*np.random.randint(5, 15), 1/(8*np.pi)
                return (a, b, c, r, s, t)
            self.gen_param = gen_param

            def f(x, param):
                a, b, c, r, s, t = param
                x1, x2 = list(x.T)
                return a*(x2-b*x1**2+c*x1-r)**2 + s*(1-t)*np.cos(x1) + s
            self.f = f
#            a = 1
#            self.b = b = 5.1/(4*np.pi**2)
#            self.c = c = 5/np.pi
#            self.r = r = 6
#            self.s = s = 10
#            self.t = t = 1/(8*np.pi)
            #self.f = f = lambda x1, x2: a*(x2-b*x1**2+c*x1-r)**2 + s*(1-t)*np.cos(x1) + s + noise*np.random.randn(*x1.shape)


        #self.fs = [lambda x: self.f(x.reshape(*input_shape), self.f_param(x)).reshape(*output_shape(x)) \
        #        + noise*np.random.randn(*output_shape(x))]
        #self.fs = [lambda x: self.f(x.reshape(*input_shape)).reshape(*output_shape(x)) \
        #        + noise*np.random.randn(*output_shape(x))]
        # batch input, output
        def fb(x, param):
            return self.f(x.reshape(*input_shape), param).reshape(*output_shape(x))
        self.fb = fb

        # noise output
        def fn(x, param):
            return self.fb(x, param) + noise*np.random.randn(*output_shape(x))
        self.fn = fn
        #self.fn = lambda x, param: self.f(x, param) + noise*np.random.randn(*output_shape(x))

        # task
        self.params = [(self.gen_param())]
        def fs(x):
            self.params[0] = (self.gen_param())
            return self.fn(x.reshape(*input_shape), self.params[0]).reshape(*output_shape(x))
        self.fs = [fs]
        #self.fs = [lambda x: self.fn(x.reshape(*input_shape), self.gen_param()).reshape(*output_shape(x))]

        for i in range(self.task_limit):
            self.params += [self.gen_param()]
            self.fs += [lambda x: self.fn(x.reshape(*input_shape), self.params[i+1]).reshape(*output_shape(x))]
        self.fs = np.array(self.fs)
        """

    def save_tasks(self, fname="task"):
        for i in range(1, self.task_limit+1):
            fname += str(i)
            task.param = self.params[i]
            task.save_task(fname)

    def load_tasks(self, fname=""):
        for i in range(1, self.task_limit+1):
            fname += str(i)
            self.params[i] = task.save_task(fname)

    def get_task_batch(self, batch_size = None, task_limit = None):
        if batch_size is None:
            batch_size = self.batch_size
        if task_limit is None:
            task_limit = self.task_limit

        return np.random.randint(0 if task_limit is 0 else 1, task_limit+1, batch_size)
        
    def get_task_idx(self, task_limit=None):
        return get_task_batch(1, task_limit)

    def get_train_test_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        xs, ys, tasks = self.generate_batch(batch_size)

        if self.disjoint_data:
            if self.random_sample:
                Ns_train = np.random.randint(self.num_samples_range[0], self.num_samples_range[1], batch_size)
                Ns_test = np.random.randint(self.num_samples_range[0], self.num_samples_range[1], batch_size)+self.num_samples_range[0]
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

        #print(len(x_train_batch[0]), len(y_train_batch[0]), len(x_test_batch[0]), len(y_test_batch[0]))
        return (x_train_batch, y_train_batch), (x_test_batch, y_test_batch), tasks

    def get_train_test_sample(self, x_y=None):
        train_batch, test_batch, tasks = self.get_train_test_batch(batch_size=1)
        x_train_batch, y_train_batch = train_batch
        x_test_batch, y_test_batch = test_batch
        #print(len(x_train_batch[0]), len(y_train_batch[0]), len(x_test_batch[0]), len(y_test_batch[0]))
        return [x_train_batch[0], y_train_batch[0]], [x_test_batch[0], y_test_batch[0]], tasks[0]

    def generate_batch(self, batch_size=None, num_samples=None):
        if batch_size is None:
            batch_size = self.batch_size
        if num_samples is None:
            num_samples = self.gen_num_samples
        
        xs = (self.input_range[:,1]-self.input_range[:,0]) * np.random.rand(batch_size, num_samples, self.io_dims[0]) + self.input_range[:,0]
        tasks = self.get_task_batch(batch_size)
        ys = map(lambda x, task: self.fs[task](x), xs, tasks)

        return xs, ys, tasks

    def generate_sample(self, num_samples=None):
        xs, ys = self.generate_batch(1)
        return xs[0], ys[0]
    
    def generate_window_samples(self, window_range=None, step_size=None, random=False):
        if window_range is None:
            window_range = self.window_range
        if step_size is None:
            step_size = self.window_step_size
        if random:
            window_range = self.window_range - np.mean(self.window_range, axis=1)
            window_range += np.random.randint(self.input_range[:,0]-window_range[:,0], self.input_range[:,1]-window_range[:,1])

        try:
            if window_range is self.window_range and \
                    step_size is self.window_step_size:
                return self.window_samples
        except AttributeError:
            r = window_range[:,1] - window_range[:,0]
            l = map(lambda w, r_: np.linspace(w[0], w[1], int(r_/step_size + 1)).reshape(-1, 1),
                    window_range, r)
            l = list(l)
            if self.io_dims[0] == 1:
                self.window_samples = l[0]
            elif self.io_dims[0] == 2:
                x1, x2 = np.meshgrid(l[0], l[1])
                self.window_samples = \
                    np.concatenate((x1.reshape(-1, 1), x2.reshape(-1, 1)), axis=1)
            return self.window_samples 

    def generate_window_samples_(self, window_range=None, step_size=None):
        if window_range is None:
            self.window_range = window_range = self.input_range
        if step_size is None:
            step_size = self.window_step_size

        r = window_range[:,1] - window_range[:,0]
        l = map(lambda w, r_: np.linspace(w[0], w[1], int(r_/step_size + 1)).reshape(-1, 1),
                window_range, r)
        l = list(l)
        if self.io_dims[0] == 1:
            self.window_samples = l[0]
        elif self.io_dims[0] == 2:
            x1, x2 = np.meshgrid(l[0], l[1])
            self.window_samples = \
                np.concatenate((x1.reshape(-1, 1), x2.reshape(-1, 1)), axis=1)
        return self.window_samples 


        if window_range is None:
            window_range = self.input_range
        if step_size is None:
            step_size = self.window_step_size


        r = window_range[:,1] - window_range[:,0]
        l = map(lambda w, r_: np.linspace(w[0], w[1], int(r_/step_size + 1)).reshape(-1, 1),
                window_range, r)
        l = list(l)
        if self.io_dims[0] == 1:
            return l[0]
        elif self.io_dims[0] == 2:
            x1, x2 = np.meshgrid(l[0], l[1])
            return np.concatenate((x1.reshape(-1, 1), x2.reshape(-1, 1)), axis=1)


        if self.io_dims[0] == 1:
            r = window_range[0][1] - window_range[0][0]
            return np.linspace(window_range[0][0], window_range[0][1], int(r[0]/step_size + 1)).reshape(-1, 1)
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
            ax.set_xlim(self.window_range[0][0], self.window_range[0][1])
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
        if self.io_dims[0] == 1:
            return np.logical_and(np.ones(len(x)), *[np.logical_and(window_range[0][0]<=t, t<=window_range[0][1]) \
                for i, t in enumerate(x.T)])
        return np.logical_and(*[np.logical_and(window_range[i][0]<=t, t<=window_range[i][1]) \
            for i, t in enumerate(x.T)])

    def plot_cov(self, ax, data, c='gray'):
        assert data.shape[1] == 3, "wrong data, need 3 aixs"
        ax.fill_between(data[:,0], 
                data[:,1] - np.sqrt(data[:,2]), 
                data[:,1] + np.sqrt(data[:,2]),
                alpha=.5, color=c2)

    def plot_data(self, ax, data, c='k'):
        data = data[self.window_crop(data[:,:self.io_dims[0]])]
        if self.io_dims == [1, 1]:
            ax.plot(data[:,0], data[:,1], color=c)
            self.plot_cov(ax, data)
        elif self.io_dims == [2, 1]:
            ax.plot_trisurf(data[:,0], data[:,1], data[:,2], color=c)
        return

        
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

    def plot_gp(self, ax, gp, data, c='c'):
        data = data[self.window_crop(data[:,:self.io_dims[0]])]
        mu, cov = gp.predict(data[:, :self.io_dims[0]], return_cov=True)
        data = np.concatenate((data, mu, cov), axis=1)
        self.plot_data(ax, data, c)
        return
        if self.io_dims == [1, 1]:
            data = data[self.window_crop(data[:,:1])]
            self.plot_data(ax, data, gp.predict)
            ax.plot(data[:,:1], gp.predict(data[:,:1]), color=c)
        elif self.io_dims == [2, 1]:
            data = data[self.window_crop(data[:,:2])]
            ax.plot_trisurf(data[:,0], data[:,1], gp.predict(data[:,:2])[:,0], color=c)

    def plot_task(self, ax, data, task=0, c='c'):
        data = data[self.window_crop(data[:,:self.io_dims[0]])]
        if 'gp' in self.datasource:
            target = self.f(data, self.params[task])
            data = np.concatenate((data, target), axis=1)
            self.plot_data(ax, data, c=c)
        elif self.datasource == 'branin':
            ax.plot_trisurf(*data[:,:self.io_dims[0]].T, 
                    self.f(data[:,0], data[:,1]), color=c)

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

