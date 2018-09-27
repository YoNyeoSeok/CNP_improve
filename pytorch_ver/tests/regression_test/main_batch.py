import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

from model import Net
#from model_batch_norm import Net
#from model_weight_norm import Net

device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 64


def loss_func(xs, mus, vars):
    normals = [Normal(mus[i], vars[i,i]) for i in range(len(xs))]
    log_probs = [normals[i].log_prob(x) for i, x in enumerate(xs)]
#    log_probs = torch.tensor([normals[i].log_prob(x) for i, x in enumerate(xs)])

#    print(log_probs)
#    loss = - torch.mean(log_probs)
#    print(loss)
    loss = 0
    for log_prob in log_probs:
        loss -= log_prob
    return loss/len(log_probs)

def main():
    x_min = -2
    x_max = 2
    x_plot = np.linspace(x_min, x_max, 51).reshape(-1, 1)
    x = np.random.rand(5)*(x_max-x_min) - (x_max-x_min)/2
    x = x.reshape(-1, 1)
    idx = np.argsort(x)
    gp = GaussianProcessRegressor(optimizer=None)
    y = gp.sample_y(x)

    gp.fit(x, y)
    y_mu, y_conv = gp.predict(x_plot, return_cov=True)

    md = Net(x.shape[1], 15, 2).to(device)
    print(md)

    optimizer = torch.optim.Adam(md.parameters(), lr=1e-1)
    

    x_train = torch.tensor(x).float().to(device)
    x_test = torch.tensor(x_plot).float().to(device)

    fig = plt.figure()
    fig.show()
    fig.canvas.draw()
    loss_bound = loss_func(torch.tensor(y_mu).float(),
        torch.tensor(y_mu.reshape(-1)).float(), y_conv)
    for i in range(1000000):
        phi = md(x_test)
        y_mu_predict = phi[:,0]
        y_cov_predict = torch.diag((phi[:,1])**2)
        loss = loss_func(torch.tensor(y_mu).float().to(device),
                    y_mu_predict, y_cov_predict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            print(i, loss.item(), 'loss_bound %.4f' % loss_bound.item())

            plt.cla()
            plt.plot(x_plot, y_mu.reshape(-1))
            plt.fill_between(x_plot.reshape(-1),
                    y_mu.reshape(-1) - np.sqrt(np.diag(y_conv)),
                    y_mu.reshape(-1) + np.sqrt(np.diag(y_conv)),
                    alpha=.5, color='k')
            plt.scatter(x,y, c='r')

            plt.plot(x_plot, y_mu_predict.cpu().data.numpy().reshape(-1))
            plt.fill_between(x_plot.reshape(-1),
                    y_mu_predict.cpu().data.numpy().reshape(-1) 
                        - np.sqrt(np.diag(y_cov_predict.cpu().data.numpy())),
                    y_mu_predict.cpu().data.numpy().reshape(-1) 
                        + np.sqrt(np.diag(y_cov_predict.cpu().data.numpy())),
                    alpha=.5, color='b')
            fig.canvas.draw()
    plt.show() 

if __name__ == '__main__':
    main()
