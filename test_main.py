"""
python main.py 
"""

from matplotlib import pyplot as plt
import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import os

from test_model import CNP_Net

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--func_limit", metavar='f', type=int, default=0,
        help='train function number, 0 is infinite (default=0)')
parser.add_argument("--max_iter", type=int, default=10000,
        help='max iteration (default=10000')
parser.add_argument("--learning_rate", metavar="lr", type=float, default=1e-4,
        help="learning rate (default=1e-4)")
parser.add_argument("--batch_size", type=int, nargs=2, default=[1,1],
        help='function batch size, observe data batch size (default: [1,1])')
parser.add_argument("--log_folder", type=str, default="log",
        help="log folder name in logs/ (default: log)")
parser.add_argument("--input_range", metavar='R', type=int, nargs=2, 
        default=[-2, 2], help="input range (default: [-2, 2])")
parser.add_argument("--fig_show", metavar='S', type=bool, default=False,
        help="figure show during traing or not (default=False)")
parser.add_argument("--load_model", type=str,
        help="load model. format: folder/iteration")
parser.add_argument("--model_layers", default=None,
        help="model layers: default=None, means {'h':[8, 32, 128], 'g':[129, 64, 32, 16, 8]}")
args = parser.parse_args()

def plot_fig(fig, x, y_min, y_cov, color='k'):
    plt.plot(x, y_min)
    plt.fill_between(x.reshape(-1), 
            y_min.reshape(-1) - np.sqrt(np.diag(y_cov)), 
            y_min.reshape(-1) + np.sqrt(np.diag(y_cov)),
            alpha=.5, color=color)



def main():
    x_min = -2
    x_max = 2 
    n_observation = 51
    x_plot = np.linspace(x_min, x_max, n_observation).reshape(-1, 1)

    if args.load_model is not None:
        model = torch.load(args.load_model)
    elif args.model_layers is not None:
        model = CNP_Net(io_dims=[1, 1], n_layers=[3, 5], layers_dim={'h':[128, 128, 128], 'g':[129, 128, 128, 128, 128]}).float()
    else:
        model = CNP_Net().float()

    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    max_iter = args.max_iter

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if args.fig_show:
        fig.show()
    fig.canvas.draw()

    length_scale = 1 
    noise = .1
    kernel = RBF(length_scale=length_scale)+WhiteKernel(noise_level=noise**2)
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)

    if args.func_limit != 0:
        xs = np.zeros((args.func_limit, n_observation))
        ys = np.zeros((args.func_limit, n_observation))
        for i in range(args.func_limit):
            xs[i] = (x_max-x_min) * (np.random.rand(n_observation) - .5)
            ys[i] = gp.sample_y(xs[i].reshape(-1, 1)).reshape(-1)
    
    batch_size = {'P': args.batch_size[0], 'N':args.batch_size[1]}

    for t in range(max_iter):
        loss = 0
        for p in range(batch_size['P']):
            # data
            if args.func_limit == 0:
                x = (x_max-x_min) * (np.random.rand(n_observation) - .5)
                x = x.reshape(-1, 1)
                y = gp.sample_y(x).reshape(-1, 1)
            else:
                i = np.random.randint(args.func_limit)
                x = xs[i].reshape(-1, 1)
                y = ys[i].reshape(-1, 1)
            idx = np.argsort(x)
           
            loss_ = 0
            for n in range(batch_size['N']):
                N = np.random.randint(1, n_observation)
           
                x_train = x[:N]
                y_train = y[:N]
                gp.fit(x_train, y_train)
                y_mu, y_cov = gp.predict(x_plot, return_cov=True)
        
                training_set = torch.cat((torch.tensor(x_train), 
                            torch.tensor(y_train)),
                        dim=1).float()
                test_set = torch.cat((torch.tensor(x).float(), 
                            torch.tensor(y).float()),
                        dim=1).float()
            
                predict_y_mu_, predict_y_cov_, log_prob = model(training_set, test_set)
        
                loss_ += -torch.sum(log_prob)
            loss += loss_ / batch_size['N']
        loss = loss / batch_size['P']

        with open("logs/%s/log.txt"%args.log_folder, "a") as log_file:
            log_file.write("%5d\t%10.4f\n"%(t, loss.item()))
        
        if t % 100 == 0:
            print('%5d'%t, '%10.4f'%loss.item())
            plt.clf()
            plt.xlim(x_min, x_max)
            plt.ylim(-3, 3)
            
            # train, test points
            plt.scatter(x, y, c='y')
            plt.scatter(x_train, y_train, c='r')

            # plot gp prediction (base line)
            plot_fig(fig, x_plot, y_mu, y_cov)
   
            # plot model prediction
            test_set = torch.cat((torch.tensor(x_plot),
                    torch.tensor(np.zeros(n_observation).reshape(-1, 1))),
                dim=1).float()
            predict_y_mu_, predict_y_cov_, _ = model(training_set, test_set)
            predict_y_mu = predict_y_mu_.data.numpy()
            predict_y_cov = np.diag(predict_y_cov_.data.numpy())**2

            plot_fig(fig, x_plot, predict_y_mu, predict_y_cov, color='b')

            fig.canvas.draw()

            plt.savefig('logs/%s/%05d.png'%(args.log_folder, t))
            torch.save(model, "logs/%s/%05d.pt"%(args.log_folder, t))
         
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    with open("logs/%s/log.txt"%args.log_folder, "a") as log_file:
        log_file.write("%5d\t%10.4f\n"%(t, loss.item()))

    t = max_iter
    print('%5d'%t, '%10.4f'%loss.item())
    plt.clf()
    plt.xlim(x_min, x_max)
    plt.ylim(-3, 3)

    # train, test points
    plt.scatter(x_train, y_train, c='r')
    plt.scatter(x, y, c='y')

    # plot gp prediction (base line)
    plot_fig(fig, x_plot, y_mu, y_cov)

    # plot model prediction
    test_set = torch.cat((torch.tensor(x_plot),
            torch.tensor(np.zeros(n_observation).reshape(-1, 1))),
        dim=1).float()
    phi, _ = model(training_set, test_set)
    predict_y_mu = phi[:,0].data.numpy()
    predict_y_cov = np.diag(phi[:,1].data.numpy())**2

    plot_fig(fig, x_plot, predict_y_mu, predict_y_cov, color='b')

    fig.canvas.draw()

    plt.savefig('logs/%s/%05d.png'%(args.log_folder, t))
    torch.save(model, "logs/%s/%05d.pt"%(args.log_folder, t))

if __name__ == '__main__':
    main()

