"""
python main.py 
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
import torch
import os, time

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
parser.add_argument("-ndd", "--not_disjoint_data", action='store_true',
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
datasource_list = ["gp1d1d", "gp2d1d", "branin"]
parser.add_argument("--datasource", type=str, nargs='?', default=datasource_list[0], choices=datasource_list,
        help="datasource list: %s"%datasource_list)
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
parser.add_argument("--time", action='store_true',
        help="time log flag")
args = parser.parse_args()

if args.gpu == -1:
    device = torch.device("cpu")
    use_cuda = False
else:
    device = torch.device("cuda:"+str(args.gpu))
    use_cuda = True

def main():
    if args.time:
        t = time.time()
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
    if args.log:
        data_generator.save_task(args.log_folder)

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
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.fig_show:
        fig = plt.figure()
        fig.show()
        ax = data_generator.make_fig_ax(fig)
        fig.canvas.draw()

    if args.time:
        print('preprocess', time.time()-t)
        t = time.time()

    for t in range(args.max_epoch):
        loss = 0
        
        train_batch, test_batch, _ = data_generator.get_train_test_batch(args.batch_size)
        x_train_batch, y_train_batch = train_batch
        x_test_batch, y_test_batch = test_batch

        if args.time:
            print('data gen', time.time()-t)
            t = time.time()

        batch_phi, batch_log_prob = zip(*map(lambda Ox, Oy, Tx, Ty:
            model(torch.cat((torch.tensor(Ox).to(device), torch.tensor(Oy).to(device)), 1),
                torch.cat((torch.tensor(Tx).to(device), torch.tensor(Ty).to(device)), 1)),
            x_train_batch, y_train_batch, x_test_batch, y_test_batch))

        loss = -sum(batch_log_prob) / args.batch_size
        
        if args.time:
            print('compute loss', time.time()-t)
            t = time.time()

        if not args.test:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if args.time:
            print('backprop', time.time()-t)
            t = time.time()

        if args.log:
            with open("logs/%s/log.txt"%args.log_folder, "a") as log_file:
                log_file.write("%5d\t%10.4f\n"%(t, loss.item()))
        
        if t % args.interval == 0:
            print('%5d'%t, '%10.4f'%loss.item())
            if args.log:
                torch.save(model, "logs/%s/%05d.pt"%(args.log_folder, t))
            if args.fig_show:
                plt.clf()
                ax = data_generator.make_fig_ax(fig)
            
                if args.random_window_position:
                    window_range = args.window_range - np.mean(args.window_range)
                    window_range += np.random.randint(args.input_range[0]-window_range[0], args.input_range[1]-window_range[1])
                    space_samples = data_generator.generate_window_samples(window_range, args.window_step_size)

                train, test, task = data_generator.get_train_test_sample()
                x_train, y_train = train
                x_test, y_test = test

                Ox, Oy = x_train, y_train
                Tx, Ty = space_samples, np.zeros((len(space_samples), data_generator.io_dims[1]))

                phi, _ = model(torch.cat((torch.tensor(Ox).to(device), torch.tensor(Oy).to(device)), 1),
                    torch.cat((torch.tensor(Tx).to(device), torch.tensor(Ty).to(device)), 1))

                if use_cuda:
                    phi = phi.cpu()

                predict_y_mu = phi[:,:data_generator.io_dims[1]].data.numpy()
                predict_y_cov = phi[:,data_generator.io_dims[1]:].data.numpy()**2
    
                train_data = np.concatenate((x_train, y_train), axis=1)
                window_data = np.concatenate((space_samples, predict_y_mu, predict_y_cov), axis=1)
                data_generator.scatter_data(ax, train_data, c='r')
                data_generator.plot_data(ax, window_data) 
                #data_generator.scatter_data(ax, test_data, c='y')
                #data_generator.contour_data(ax, window_data) 
                if not args.test:
                    #gp = data_generator.gp().fit(train_data[:data_generator.io_dims[0], data_generator.io_dims[0]:])
                    gp = data_generator.gp().fit(x_train, y_train)
                    data_generator.plot_gp(ax, gp, space_samples)
                else:
                    data_generator.plot_task(ax, space_samples, args.task_limit)

                title = args.log_folder + "/step_" + str(t) + "/points_" + str(len(x_train))
                if args.task_limit != 0:
                    title += "/task_" + str(task)
                data_generator.make_fig_title(fig, title)
                fig.canvas.draw()

                if args.log:
                    plt.savefig('logs/%s/%05d.png'%(args.log_folder, t))
        if args.time:
            print('log time', time.time()-t)
            t = time.time()
    if args.log:
        torch.save(model, "logs/%s/%05d.pt"%(args.log_folder, args.max_epoch-1))
if __name__ == '__main__':
    main()

