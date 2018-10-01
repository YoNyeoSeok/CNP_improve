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
parser.add_argument("--window_range", metavar=('min', 'max'), type=int, nargs=2, default=None,
        help='plot window range (default: input_range)')
parser.add_argument("--window_step_size", type=float, nargs='?', default=1.0,
        help='plot window step size (default: 1.0)')
parser.add_argument("--random_window_position", action='store_true',
        help='plot window position by random in input range (default: false)')
parser.add_argument("--log", action='store_true',
        help="save loss, fig and model log")
parser.add_argument("--log_folder", type=str, default="log",
        help="log folder name in logs/ (default: log)")
parser.add_argument("--datasource", type=str, nargs='?', default="gp1d", choices=["gp1d", "branin"],
        help="gp1d or branin")
parser.add_argument("--fig_show", action='store_true',
        help="figure show during traing")
parser.add_argument("--gpu", type=int, nargs='?', default=-1,
        help="use gpu")
parser.add_argument("--load_model", type=str,
        help="load model. format: folder/iteration")
parser.add_argument("--model_layers", nargs='?', default=None,
        help="model layers: default=None, means {'h':[8, 32, 128], 'g':[128, 64, 32, 16, 8]}")
parser.add_argument("--test", action='store_true',
        help="test flag (not train)")
args = parser.parse_args()

if args.gpu == -1:
    device = torch.device("cpu")
    use_cuda = False
else:
    device = torch.device("cuda:"+str(args.gpu))
    use_cuda = True

def train(data_generator, model, optimizer, args):
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
    
            training_set = torch.cat((torch.tensor(x_train),
                        torch.tensor(y_train)),
                    dim=1)
            test_set = torch.cat((torch.tensor(x_test),
                        torch.tensor(y_test)),
                    dim=1)
            #training_set.float()
            #test_set.float()
            training_set.double()
            test_set.double()
        
            if use_cuda:
                training_set = training_set.to(device)
                test_set = test_set.to(device)
            #print(training_set.shape, test_set.shape)
            # print('train, test', training_set.shape, test_set.shape)
            phi, log_prob = model(training_set, test_set)
            # print('phi', phi.shape)
    
            loss += -torch.sum(log_prob)
        loss = loss / args.batch_size

        if args.log:
            with open("logs/%s/log.txt"%args.log_folder, "a") as log_file:
                log_file.write("%5d\t%10.4f\n"%(t, loss.item()))
        
        if t % args.interval == 0:
            print('%5d'%t, '%10.4f'%loss.item())
            if args.fig_show:
                plt.clf()
                ax = data_generator.make_fig_ax(fig)
            
            # train, test points
            # print(x_test.shape, y_test.shape)
            # data_generator.plot_data(fig, np.concatenate((x_test, y_test), axis=1))
            # if args.fig_show:

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
                if args.random_window_position:
                    window_range = args.window_range - np.mean(args.window_range)
                    window_range += np.random.randint(args.input_range[0]-window_range[0], args.input_range[1]-window_range[1])
                    space_samples = data_generator.generate_window_samples(window_range, args.window_step_size)

                test_set = torch.cat((torch.tensor(space_samples),
                        torch.tensor(np.zeros(len(space_samples)).reshape(-1, 1))),
                    dim=1)
                #test_set.float()
                test_set.double()
                if use_cuda:
                    test_set = test_set.to(device)
                    print('before forward current gpu memory usage', torch.cuda.memory_allocated(torch.cuda.current_device()))
                # print('train, test', training_set.shape, test_set.shape)
                phi, _ = model(training_set, test_set)
                if use_cuda:
                    print('after forward current gpu memory usage', torch.cuda.memory_allocated(torch.cuda.current_device()))
                if use_cuda:
                    phi = phi.cpu()
                # print('phi', phi.shape)
                predict_y_mu = phi[:,:data_generator.io_dims[1]].data.numpy()
                predict_y_cov = phi[:,data_generator.io_dims[1]:].data.numpy()**2
    #            predict_y_mu_, predict_y_cov_, _ = model(training_set, test_set)
    #            predict_y_mu = predict_y_mu_.data.numpy()
    #            predict_y_cov = np.diag(predict_y_cov_.data.numpy())**2
    
                # plot_fig(fig, x_plot, predict_y_mu, predict_y_cov, color='b')
                # print(space_samples.shape, predict_y_mu.shape, predict_y_cov.shape)
                #test_data = np.concatenate((x_test, y_test), axis=1)
                train_data = np.concatenate((x_train, y_train), axis=1)
                window_data = np.concatenate((space_samples, predict_y_mu, predict_y_cov), axis=1)
                data_generator.scatter_data(ax, train_data, c='r')
                data_generator.plot_data(ax, window_data) 
                #data_generator.scatter_data(ax, test_data, c='y')
                #data_generator.contour_data(ax, window_data) 
                #data_generator.plot_gp(ax, train_data, window_data)
                fig.canvas.draw()

            if args.log:
                plt.savefig('logs/%s/%05d.png'%(args.log_folder, t))
                torch.save(model, "logs/%s/%05d.pt"%(args.log_folder, t))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(data_generator, model, args):
    pass

def main():
    data_generator = DataGenerator(datasource=args.datasource, 
                                   batch_size=args.batch_size,
                                   random_sample=args.random_sample,
                                   disjoint_data=not args.not_disjoint_data,
                                   num_samples=args.num_samples,
                                   num_samples_range=args.num_samples_range,
                                   input_range=args.input_range,
                                   window_range=args.window_range,
                                   window_step_size=args.window_step_size,
                                   random_window_position=args.random_window_position,
                                   task_limit=args.task_limit)
    
    if not args.random_window_position:
        space_samples = data_generator.generate_window_samples(args.window_range, args.window_step_size)

    if args.load_model is not None:
        model = torch.load(args.load_model)
    elif args.model_layers is not None:
        model = CNP_Net(io_dims=data_generator.io_dims, 
                        layers_dim={'h':[8, 32, 128], 
                                    'g':[128, 64, 32, 16, 8]})#.float()
    else:
        model = CNP_Net(io_dims=data_generator.io_dims)#.float()

    model.double()

    if use_cuda:
        model.to(device)
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
    
            training_set = torch.cat((torch.tensor(x_train),
                        torch.tensor(y_train)),
                    dim=1)
            test_set = torch.cat((torch.tensor(x_test),
                        torch.tensor(y_test)),
                    dim=1)
            #training_set.float()
            #test_set.float()
            training_set.double()
            test_set.double()
        
            if use_cuda:
                training_set = training_set.to(device)
                test_set = test_set.to(device)
            #print(training_set.shape, test_set.shape)
            # print('train, test', training_set.shape, test_set.shape)
            phi, log_prob = model(training_set, test_set)
            # print('phi', phi.shape)
    
            loss += -torch.sum(log_prob)
        loss = loss / args.batch_size

        if args.log:
            with open("logs/%s/log.txt"%args.log_folder, "a") as log_file:
                log_file.write("%5d\t%10.4f\n"%(t, loss.item()))
        
        if t % args.interval == 0:
            print('%5d'%t, '%10.4f'%loss.item())
            if args.fig_show:
                plt.clf()
                ax = data_generator.make_fig_ax(fig)
            
            # train, test points
            # print(x_test.shape, y_test.shape)
            # data_generator.plot_data(fig, np.concatenate((x_test, y_test), axis=1))
            # if args.fig_show:

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
                if args.random_window_position:
                    window_range = args.window_range - np.mean(args.window_range)
                    window_range += np.random.randint(args.input_range[0]-window_range[0], args.input_range[1]-window_range[1])
                    space_samples = data_generator.generate_window_samples(window_range, args.window_step_size)

                test_set = torch.cat((torch.tensor(space_samples),
                        torch.tensor(np.zeros(len(space_samples)).reshape(-1, 1))),
                    dim=1)
                #test_set.float()
                test_set.double()
                if use_cuda:
                    test_set = test_set.to(device)
                    #print('before forward current gpu memory usage', torch.cuda.memory_allocated(torch.cuda.current_device()))
                # print('train, test', training_set.shape, test_set.shape)
                phi, _ = model(training_set, test_set)
                #if use_cuda:
                    #print('after forward current gpu memory usage', torch.cuda.memory_allocated(torch.cuda.current_device()))
                if use_cuda:
                    phi = phi.cpu()
                # print('phi', phi.shape)
                predict_y_mu = phi[:,:data_generator.io_dims[1]].data.numpy()
                predict_y_cov = phi[:,data_generator.io_dims[1]:].data.numpy()**2
    #            predict_y_mu_, predict_y_cov_, _ = model(training_set, test_set)
    #            predict_y_mu = predict_y_mu_.data.numpy()
    #            predict_y_cov = np.diag(predict_y_cov_.data.numpy())**2
    
                # plot_fig(fig, x_plot, predict_y_mu, predict_y_cov, color='b')
                # print(space_samples.shape, predict_y_mu.shape, predict_y_cov.shape)
                #test_data = np.concatenate((x_test, y_test), axis=1)
                train_data = np.concatenate((x_train, y_train), axis=1)
                window_data = np.concatenate((space_samples, predict_y_mu, predict_y_cov), axis=1)
                data_generator.scatter_data(ax, train_data, c='r')
                data_generator.plot_data(ax, window_data) 
                #data_generator.scatter_data(ax, test_data, c='y')
                #data_generator.contour_data(ax, window_data) 
                #data_generator.plot_gp(ax, train_data, window_data)
                fig.canvas.draw()

            if args.log:
                plt.savefig('logs/%s/%05d.png'%(args.log_folder, t))
                torch.save(model, "logs/%s/%05d.pt"%(args.log_folder, t))

    if not args.test:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
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

