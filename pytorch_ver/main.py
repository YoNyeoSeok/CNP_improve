"""
python main.py 
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
import torch
import os

from model import CNP_Net
from data_generator import DataGenerator

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task_limit", type=int, default=0,
        help='number of task to train, 0 is infinite (default=0)')
parser.add_argument("--max_epoch", type=int, default=10000,
        help='max iteration (default=10000')
parser.add_argument("--interval", type=int, default=1000,
        help='max iteration (default=10000')
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4,
        help="learning rate (default=1e-4)")
parser.add_argument("-bs", "--batch_size", type=int, nargs='?', default=32,
        help='batch size (default: 32)')
parser.add_argument("-ns", "--num_samples", type=int, nargs=2, default=[5, 5],
        help='number of samples, few-shot number of train, test (default: [5, 5])')
parser.add_argument("-nsr", "--num_samples_range", metavar=('min', 'max'), type=int, nargs=2, default=[1, 51],
        help='number of samples random range, few-shot number random range (default: [1 51])')
parser.add_argument("-rs", "--random_sample", action='store_true',
        help='number of samples is random flag')
parser.add_argument("-nnd", "--not_disjoint_data", action='store_false',
        help='train test set data are not disjoint setting flag')
parser.add_argument("--input_range", metavar=('min', 'max'), type=int, nargs=2, default=[-2, 2],
        help='function input range (default: [-2, 2])')
parser.add_argument("--output_range", metavar=('min', 'max'), type=int, nargs=2, default=[-2, 2],
        help='function output range (default: [-2, 2])')
parser.add_argument("--log", action='store_true',
        help="save loss, fig and model log")
parser.add_argument("--log_folder", type=str, default="log",
        help="log folder name in logs/ (default: log)")
parser.add_argument("--datasource", type=str, nargs='?', default="gp1d", choices=["gp1d", "branin"],
        help="gp1d or branin")
parser.add_argument("--fig_show", action='store_true',
        help="figure show during traing")
parser.add_argument("--gpu", action='store_false',
        help="use gpu")
parser.add_argument("--load_model", type=str,
        help="load model. format: folder/iteration")
parser.add_argument("--model_layers", nargs='?', default=None,
        help="model layers: default=None, means {'h':[8, 32, 128], 'g':[128, 64, 32, 16, 8]}")
parser.add_argument("--precision", nargs='?', default=None,
        help="change variance to precision (inverse) option")
args = parser.parse_args()

if args.gpu:
    device = torch.device("cuda:0")

def plot_fig(fig, x, y_min, y_cov, color='k'):
    plt.plot(x, y_min)
    plt.fill_between(x.reshape(-1), 
            y_min.reshape(-1) - np.sqrt(np.diag(y_cov)), 
            y_min.reshape(-1) + np.sqrt(np.diag(y_cov)),
            alpha=.5, color=color)

def main():
    data_generator = DataGenerator(datasource=args.datasource, 
                                   batch_size=args.batch_size,
                                   random_sample=args.random_sample,
                                   disjoint_data=not args.not_disjoint_data,
                                   num_samples=args.num_samples,
                                   num_samples_range=args.num_samples_range,
                                   input_range=args.input_range,
                                   output_range=args.output_range,
                                   task_limit=args.task_limit)
    
    space_samples = data_generator.generate_space_sample()
#    x_plot = np.linspace(args.input_range[0], args.input_range[1], args.num_samples[-1]).reshape(-1, 1)

    # x_plot = np.linspace(x_min, x_max, n_observation).reshape(-1, 1)

    if args.load_model is not None:
        model = torch.load(args.load_model)
    elif args.model_layers is not None:
        model = CNP_Net(io_dims=data_generator.io_dims, 
                        layers_dim={'h':[8, 32, 128], 
                                    'g':[128, 64, 32, 16, 8]})#.float()
    else:
        model = CNP_Net(io_dims=data_generator.io_dims)#.float()

    if args.gpu:
        model.to(device)
        print('max gpu memory', torch.cuda.max_memory_allocated(torch.cuda.current_device()))
        print('current gpu memory usage', torch.cuda.memory_allocated(torch.cuda.current_device()))
#    for m, p in model.named_parameters():
#        print(m, p)
#    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.fig_show:
        fig = plt.figure()
        fig.show()
        ax = data_generator.make_fig_ax(fig)
        fig.canvas.draw()

    for t in range(args.max_epoch):
        # print(t)
        loss = 0
        #x, y = data_generator.generate_batch()
        for p in range(args.batch_size):
            # data
            # x, y = data_generator.generate_sample()
            train, test = data_generator.get_train_test_sample()
            # print('train shape', train)
            # print('test shape', test)
            x_train, y_train = train
            x_test, y_test = test

            # print(x_train, y_train, x_test, y_test)
            # print('shapes', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
            # print(x_test.shape, y_test.shape, x_train.shape, y_train.shape)
            
#            if self.num_samples == 0:
#                N = np.random.randint(self.num_samples_min, self.num_samples_max)
#            else:
#                N = self.num_samples
#            if args.datasource == 'gp1d':
#                gp = data_generator.gp
#                gp.fit(x_train, y_train)
#                #y_mu, y_cov = gp.predict(x_plot, return_cov=True)
#                y_mu, y_cov = gp.predict(space_samples, return_cov=True)
#            elif args.datasource == 'branin':
#                pass

            # print('train', x_train.shape, y_train.shape)
    
            training_set = torch.cat((torch.tensor(x_train),
                        torch.tensor(y_train)),
                    dim=1).float()
            test_set = torch.cat((torch.tensor(x_test),
                        torch.tensor(y_test)),
                    dim=1).float()
        
            if args.gpu:
                training_set = training_set.to(device)
                test_set = test_set.to(device)
                print('before forward current gpu memory usage', torch.cuda.memory_allocated(torch.cuda.current_device()))
            print(training_set.shape, test_set.shape)
            # print('train, test', training_set.shape, test_set.shape)
            phi, log_prob = model(training_set, test_set)
            # print('phi', phi.shape)
            if args.gpu:
                print('after forward current gpu memory usage', torch.cuda.memory_allocated(torch.cuda.current_device()))
    
            loss += -torch.sum(log_prob)
        loss = loss / args.batch_size

        if args.log:
            with open("logs/%s/log.txt"%args.log_folder, "a") as log_file:
                log_file.write("%5d\t%10.4f\n"%(t, loss.item()))
        
        if (t+1) % args.interval == 0:
            print('%5d'%t, '%10.4f'%loss.item())
            if args.fig_show:
                plt.clf()
                ax = data_generator.make_fig_ax(fig)
            
            # train, test points
            # print(x_test.shape, y_test.shape)
            # data_generator.plot_data(fig, np.concatenate((x_test, y_test), axis=1))
            if args.fig_show:
                data_generator.scatter_data(ax, np.concatenate((x_test, y_test), axis=1), c='y')
                data_generator.scatter_data(ax, np.concatenate((x_train, y_train), axis=1), c='r')

    #                if x_test.shape[1] == 1:
    #                    plt.scatter(x_test, y_test, c='y')
    #                    plt.scatter(x_train, y_train, c='r')
    #            
    #                    # plot gp prediction (base line)
    #                    plot_fig(fig, x_plot, y_mu, y_cov)
       
                # plot model prediction
    #            print(x_plot.shape)
    #            test_set = torch.cat((torch.tensor(x_plot),
    #                    torch.tensor(np.zeros(len(x_plot)).reshape(-1, 1))),
    #                dim=1).float()
                # print(space_samples)
                test_set = torch.cat((torch.tensor(space_samples),
                        torch.tensor(np.zeros(len(space_samples)).reshape(-1, 1))),
                    dim=1).float()
                if args.gpu:
                    test_set = test_set.to(device)
                # print('train, test', training_set.shape, test_set.shape)
                phi, _ = model(training_set, test_set)
                if args.gpu:
                    phi = phi.cpu()
                # print('phi', phi.shape)
                predict_y_mu = phi[:,:data_generator.io_dims[1]].data.numpy()
                predict_y_cov = phi[:,data_generator.io_dims[1]:].data.numpy()**2
    #            predict_y_mu_, predict_y_cov_, _ = model(training_set, test_set)
    #            predict_y_mu = predict_y_mu_.data.numpy()
    #            predict_y_cov = np.diag(predict_y_cov_.data.numpy())**2
    
                # plot_fig(fig, x_plot, predict_y_mu, predict_y_cov, color='b')
                # print(space_samples.shape, predict_y_mu.shape, predict_y_cov.shape)
                data_generator.plot_data(ax, np.concatenate((space_samples, predict_y_mu, predict_y_cov), axis=1))
                fig.canvas.draw()

            if args.log:
                plt.savefig('logs/%s/%05d.png'%(args.log_folder, t))
                torch.save(model, "logs/%s/%05d.pt"%(args.log_folder, t))
        print('before backward') 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('before backward') 
    
#    if args.log:
#        with open("logs/%s/log.txt"%args.log_folder, "a") as log_file:
#            log_file.write("%5d\t%10.4f\n"%(t, loss.item()))
#
##    t = args.max_epoch
#    print('%5d'%t, '%10.4f'%loss.item())
#    if args.fig_show:
#        test_set = torch.cat((torch.tensor(space_samples),
#                torch.tensor(np.zeros(len(space_samples)).reshape(-1, 1))),
#            dim=1).float()
#        phi, _ = model(training_set, test_set)
#        predict_y_mu = phi[:,:data_generator.io_dims[1]].data.numpy()
#        predict_y_cov = phi[:,data_generator.io_dims[1]:].data.numpy()**2
#    
#        plt.clf()
#        ax = data_generator.make_fig_ax(fig)
#        data_generator.scatter_data(ax, np.concatenate((x_train, y_train), axis=1), c='r')
#        data_generator.scatter_data(ax, np.concatenate((x_test, y_test), axis=1), c='y')
#        data_generator.plot_data(ax, np.concatenate((space_samples, predict_y_mu, predict_y_cov), axis=1))
#        fig.canvas.draw()
#        fig.show()
##    plt.clf()
##    plt.xlim(x_min, x_max)
##    plt.ylim(-3, 3)
#
#    # train, test points
##    plt.scatter(x_train, y_train, c='r')
##    plt.scatter(x, y, c='y')
#
#    # plot gp prediction (base line)
##    plot_fig(fig, x_plot, y_mu, y_cov)
#
#    # plot model prediction
##    test_set = torch.cat((torch.tensor(x_plot),
##            torch.tensor(np.zeros(n_observation).reshape(-1, 1))),
##        dim=1).float()
##    phi, _ = model(training_set, test_set)
##    predict_y_mu = phi[:,0].data.numpy()
##    predict_y_cov = np.diag(phi[:,1].data.numpy())**2
##
##    plot_fig(fig, x_plot, predict_y_mu, predict_y_cov, color='b')
#
##    fig.canvas.draw()
#
#    if args.log:
#        torch.save(model, "logs/%s/%05d.pt"%(args.log_folder, t))
#        if args.fig_show:
#            plt.savefig('logs/%s/%05d.png'%(args.log_folder, t))

if __name__ == '__main__':
    main()

