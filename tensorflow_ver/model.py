import tensorflow as tf
import numpy as np
import scipy
from scipy.linalg import cholesky

class Net_h():
    def __init__(self, dim_input, dim_hiddens):
        self.dim_input = dim_input 
        self.dim_hiddens = dim_hiddens

    ## trian, test flow    
    def construct_model(self, O):
         
        pass

    ## initial weights
    def construct_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hiddens[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros(self.dim_hiddens[0]))
        for i in range(1, len(self.dim_hiddens)):
            weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([self.dim_hiddens[i-1], self.dim_hiddens[i]], stddev=0.01))
            weights['b'+str(i+1)] = tf.Variable(tf.zeros(self.dim_hiddens[i]))
        return weights

    ## forward path
    def forward(self, inp, weights):
        outp = tf.matmul(inp, weights['w1']) + weights['b1']
        acti = tf.nn.relu(outp)
        for i in range(1, len(self.dim_hiddens)):
            outp = tf.matmul(acti, weights['w'+str(i+1)]) + weights['b'+str(i+1)]
            acti = tf.nn.relu(outp)
        return acti

class Net_g():
    def __init__(self, dim_hiddens, dim_output):
        self.dim_hiddens = dim_hiddens
        self.dim_output = dim_output 

    ## trian, test flow    
    def construct_model(self, O):
         
        pass

    ## initial weights
    def construct_weights(self):
        weights = {}
        for i in range(len(self.dim_hiddens)-1):
            weights['w'+str(i)] = tf.Variable(tf.truncated_normal([self.dim_hiddens[i], self.dim_hiddens[i+1]], stddev=0.01))
            weights['b'+str(i)] = tf.Variable(tf.zeros(self.dim_hiddens[i+1]))
        weights['w'+str(len(self.dim_hiddens))] = tf.Variable(tf.truncated_normal([self.dim_hiddens[str(len(self.dim_hiddens)-1)], self.dim_output], stddev=0.01))
        weights['b'+str(len(self.dim_hiddens))] = tf.Variable(tf.zeros(self.dim_output))
        return weights

    ## forward path
    def forward(self, inp, weights):
        for i in range(len(self.dim_hiddens)):
            outp = tf.matmul(acti, weights['w'+str(i)]) + weights['b'+str(i)]
            acti = tf.nn.relu(outp)
        outp = tf.matmul(inp, weights['w'+str(len(self.dim_hiddens))]) + weights['b'+str(len(self.dim_hiddens))]
        acti = tf.nn.relu(outp)
        return acti

class CNP_Net():
    def __init__(self, dim_io=[1,1], \
            dim_hiddens={'h':[8, 32, 128], 'g':[128, 64, 32, 16, 8]}):
        self.dim_io = dim_io 
        self.dim_hiddens = dim_hiddens
        self.net_h = Net_h(sum(dim_io), dim_hiddens['h'])
        self.net_g = Net_g(dim_hiddens['g'], dim_io[1]**2)

    def construct_model(self):
        pass
    
    def construct_weights(self):
        weights_g = self.net_g.construct_weights()
        weights_h = self.net_h.construct_weights()
        weights = {}
        for keys_g, keys_h in zip(weights_g, weights_h):
            weights['g'+keys_g] = weights_g.pop(keys_g)
            weights['h'+keys_h] = weights_h.pop(keys_h)
        del weights_g, weights_h

    def forward(self, O, T, weights_g, weights_h):
        import time
        t = time.time()
        self.rs = self.net_h.forward(O, weights_h)
        """
        self.r = self.operator(self.net_h(O), dim=0).expand(T.shape[0], -1)
        self.xr = torch.cat((self.r, T[:,:self.io_dims[0]]), dim=1)
        self.phi = self.net_g(self.xr)

#        if self.io_dims[1] != 1: 
#            print("multivariate regression is not supported")
#            return
        t = time.time()
        self.mu = self.phi[:,:self.io_dims[1]]
        self.sig = self.softplus(self.phi[:,self.io_dims[1]:])
        
        t = time.time()
        log_probs = []
        def func(m, s, t):
            normal = MultivariateNormal(m, torch.diag(s**2))
            return normal.log_prob(t)
#        print(self.mu)
#        print(self.sig)
#        print(T)
#        print(T[:, self.io_dims[0]:])
        log_probs = list(map(func, self.mu, self.sig, T[:, self.io_dims[0]:]))
#        for m, s, t in zip(self.mu, self.sig, T):
#            #            normal = MultivariateNormal(m, torch.diag(s**2))
##            log_probs.append(normal.log_prob(t[self.io_dims[0]:]))
#            print(m)
#            log_probs.append(t.log_normal_(m[0].cpu().numpy().tolist(), s[0].cpu().numpy().tolist()**2))
#            log_probs.append(normal.log_prob(t[self.io_dims[0]:]))

    
#        t = time.time()
#        normals = [MultivariateNormal(mu, torch.diag(cov)) for mu, cov in 
#                zip(self.mu, self.sig**2)]
#        print('normals', time.time() - t)
#        t = time.time()
#        log_probs = [normals[i].log_prob(t[self.io_dims[0]:]) for i, t in enumerate(T)]
#        print('log_probs', time.time() - t)

        log_prob = 0
        for p in log_probs:
            log_prob += p
        
        return self.phi, log_prob/len(log_probs)
        """
"""
class net_h(nn.module):
    def __init__(self, input_dim, layers_dim):
        super(net_h, self).__init__()
        net_h = [nn.linear(input_dim, layers_dim[0])]
        net_h_ = [nn.linear(layers_dim[i-1], layers_dim[i])
                  for i in range(1, len(layers_dim))]
        
        self.net = nn.modulelist(np.concatenate((net_h, net_h_)))

    def forward(self, x):
        for i, l in enumerate(self.net):
            x = l(x)
            if i+1 < len(self.net):
                x = f.relu(x)
#                x = f.sigmoid(x)
        return x
    
        
class net_g(nn.module):
    def __init__(self, layers_dim, io_dims):
        super(net_g, self).__init__()
        layers_dim[0] += io_dims[0]
        net_g = [nn.linear(layers_dim[i-1], layers_dim[i]) 
                 for i in range(1, len(layers_dim))]
        net_g_ = [nn.linear(layers_dim[-1], io_dims[1]*2)]
        
        self.net = nn.modulelist(np.concatenate((net_g, net_g_)))

    def forward(self, x):
        for i, l in enumerate(self.net):
            x = l(x)
            if i+1 < len(self.net):
                x = f.relu(x)
#                x = f.sigmoid(x)
        return x 
    
    
class CNP_Net(nn.module):
    def __init__(self, io_dims=[1,1], \
                 layers_dim={'h':[8, 32, 128], 'g':[128, 64, 32, 16, 8]}):
        super(cnp_net, self).__init__()
        self.io_dims = io_dims

        self.net_h = Net_h(np.sum(io_dims), layers_dim['h'])
        self.net_g = Net_g(layers_dim['g'], io_dims)
        self.operator = torch.mean
        self.softplus = nn.Softplus()

    def forward(self, O, T):
        import time
        t = time.time()
        self.r = self.operator(self.net_h(O), dim=0).expand(T.shape[0], -1)
        self.xr = torch.cat((self.r, T[:,:self.io_dims[0]]), dim=1)
        self.phi = self.net_g(self.xr)

#        if self.io_dims[1] != 1: 
#            print("multivariate regression is not supported")
#            return
        t = time.time()
        self.mu = self.phi[:,:self.io_dims[1]]
        self.sig = self.softplus(self.phi[:,self.io_dims[1]:])
        
        t = time.time()
        log_probs = []
        def func(m, s, t):
            normal = MultivariateNormal(m, torch.diag(s**2))
            return normal.log_prob(t)
#        print(self.mu)
#        print(self.sig)
#        print(T)
#        print(T[:, self.io_dims[0]:])
        log_probs = list(map(func, self.mu, self.sig, T[:, self.io_dims[0]:]))
#        for m, s, t in zip(self.mu, self.sig, T):
#            #            normal = MultivariateNormal(m, torch.diag(s**2))
##            log_probs.append(normal.log_prob(t[self.io_dims[0]:]))
#            print(m)
#            log_probs.append(t.log_normal_(m[0].cpu().numpy().tolist(), s[0].cpu().numpy().tolist()**2))
#            log_probs.append(normal.log_prob(t[self.io_dims[0]:]))

    
#        t = time.time()
#        normals = [MultivariateNormal(mu, torch.diag(cov)) for mu, cov in 
#                zip(self.mu, self.sig**2)]
#        print('normals', time.time() - t)
#        t = time.time()
#        log_probs = [normals[i].log_prob(t[self.io_dims[0]:]) for i, t in enumerate(T)]
#        print('log_probs', time.time() - t)

        log_prob = 0
        for p in log_probs:
            log_prob += p
        
        return self.phi, log_prob/len(log_probs)
"""
