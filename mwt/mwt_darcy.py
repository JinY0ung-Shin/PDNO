import torch
import torch.nn as nn

import numpy as np
from scipy.io import loadmat, savemat
import math
import os
import h5py

from functools import partial
from models.models import MWT2d
from models.utils import train, test, LpLoss, get_filter, UnitGaussianNormalizer
from tqdm import tqdm
import argparse

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_initializer(name):
    
    if name == 'xavier_normal':
        init_ = partial(nn.init.xavier_normal_)
    elif name == 'kaiming_uniform':
        init_ = partial(nn.init.kaiming_uniform_)
    elif name == 'kaiming_normal':
        init_ = partial(nn.init.kaiming_normal_)
    return init_

def get_args(argv=None):
    parser = argparse.ArgumentParser(description = 'Put your hyperparameters')
    parser.add_argument('name', type=str, help='experiments name')
    parser.add_argument('--batch', default=10, type=int, help = 'batch size')
    parser.add_argument('--epochs', default=500, type=int, help = 'Number of Epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--step_size', default=100, type=int, help='scheduler step size')
    parser.add_argument('--gamma', default=0.5, type=float, help='scheduler factor')
    parser.add_argument('--res', default=16, type=int, help='resolution parametr')
    
    return parser.parse_args(argv)


if __name__=='__main__':
    args = get_args()
    print(args)    
    
    
    ntrain = 1000
    ntest = 200
    r = args.res
    h = int(((512 - 1)/r) + 1)
    s = h
    #s=64
    print(h)
    
    dataloader1 = torch.load('../data/darcy_r512_N1024.bin')
    dataloader2 = torch.load('../data/darcy_r512_N1024.bin')
    a_tensor = torch.cat([dataloader1['coeff'], dataloader2['coeff']], dim=0).float()
    p_tensor = torch.cat([dataloader1['sol'], dataloader2['sol']], dim=0).float()
    
    x_train = a_tensor[:ntrain,::r,::r][:,:s,:s]
    y_train = p_tensor[:ntrain,::r,::r][:,:s,:s]

    x_test = a_tensor[-ntest:,::r,::r][:,:s,:s]
    y_test = p_tensor[-ntest:,::r,::r][:,:s,:s]
    del a_tensor, p_tensor
    
    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)

    grids = []
    grids.append(np.linspace(0, 1, s))
    grids.append(np.linspace(0, 1, s))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    grid = grid.reshape(1,s,s,2)
    grid = torch.tensor(grid, dtype=torch.float)
    x_train = torch.cat([x_train.reshape(ntrain,s,s,1), grid.repeat(ntrain,1,1,1)], dim=3)
    x_test = torch.cat([x_test.reshape(ntest,s,s,1), grid.repeat(ntest,1,1,1)], dim=3)
    
    batch_size = args.batch
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    ich = 3
    initializer = get_initializer('xavier_normal') # xavier_normal, kaiming_normal, kaiming_uniform
    model = MWT2d(ich, 
                alpha = 12,
                c = 4,
                k = 4, 
                base = 'legendre', # 'chebyshev'
                nCZ = 4,
                L = 0,
                initializer = initializer,
                ).to(device)
    learning_rate = args.lr

    epochs = args.epochs
    step_size = args.step_size
    gamma = args.gamma
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    myloss = LpLoss(size_average=False)
    y_normalizer.cuda()

    train_loss = []
    test_loss = []
    for epoch in tqdm(range(1, epochs+1)):
        train_l2 = train(model, train_loader, optimizer, epoch, device,
            lossFn = myloss, lr_schedule = scheduler,
            post_proc = y_normalizer.decode)

        test_l2 = test(model, test_loader, device, lossFn=myloss, post_proc=y_normalizer.decode)
        print(f'epoch: {epoch}, train l2 = {train_l2}, test l2 = {test_l2}')
        
        train_loss.append(train_l2)
        test_loss.append(test_l2)
    torch.save({'train_rel':train_loss, 'test_rel':test_loss}, 'results/loss_{}.bin'.format(r))