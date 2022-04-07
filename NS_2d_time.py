import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from utils import MatReader, rel_error

from tqdm import tqdm
import sys
import argparse
import os
import shutil

torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self, d_in, d_out, act, num_layer=2, num_hidden=64):
        super().__init__()
        self.linear_in = nn.Linear(d_in, num_hidden)
        self.hidden = nn.ModuleList([nn.Linear(num_hidden, num_hidden) for i in range(num_layer)])
        self.linear_out = nn.Linear(num_hidden, d_out)
        act = act.lower()
        if act=='tanh':
            self.activation=torch.tanh
        if act=='gelu':
            self.activation = F.gelu
            
    
    def forward(self, x):
        out = self.linear_in(x)
        out = F.gelu(out)
        for layer in self.hidden:
            out = layer(out)
            out = self.activation(out)
        return self.linear_out(out)
    
class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, num_layer, num_hidden, symbol_act, net_out):
        super(SpectralConv2d_fast, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net_out = net_out

        self.scale = (1 / (in_channels * out_channels))
        self.net_real = Net(d_in=2, d_out=in_channels*out_channels, num_layer=num_layer, num_hidden=num_hidden, act= symbol_act)
        self.net_imag= Net(d_in=2, d_out=in_channels*out_channels, num_layer=num_layer, num_hidden=num_hidden, act= symbol_act)
        if net_out:
            self.net_out = Net(d_in=2, d_out=in_channels*out_channels, num_layer=num_layer, 
                               num_hidden=num_hidden, act= symbol_act)
    
    def _weights(self, shape):
        grid = self.get_grid_freq(shape)
        out_real = self.net_real(grid).permute(2, 0, 1).contiguous()
        out_imag = self.net_imag(grid).permute(2, 0, 1).contiguous()
        out_real = out_real.reshape(self.out_channels, self.in_channels, *(grid.shape[:2]))
        out_imag = out_imag.reshape(self.out_channels, self.in_channels, *(grid.shape[:2]))
        return torch.complex(out_real, out_imag)
    
    def _weights_out(self, shape):
        grid = self.get_grid(shape)
        out = self.net_out(grid).permute(2, 0, 1).contiguous()
        out = out.reshape(self.out_channels, self.in_channels, shape[0], shape[1])
        return out
    
    def cal_weights(self, shape):
        self.set_shape(shape)
        self.weights = self._weights(shape)
        if self.net_out:
            self.weights_out = self._weights_out(shape)
        
    def set_shape(self, shape):
        self.shape = shape

    def forward(self, x):
        batchsize = x.shape[0]
        shape = x.shape[-2:]
        self.cal_weights(shape)
        
        x_ft = torch.fft.rfft2(x) # (B, 1, 20, 64, 64)
        
        out_ft = (x_ft.unsqueeze(dim=1))*self.weights
        if self.net_out:
            x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
            x = x*self.weights_out
            x = x.sum(dim=2)
        else:
            out_ft = out_ft.sum(dim=2) # (B, 20, 64, 64)
            x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))        
        return x
    
    def get_grid(self, shape):
        mx, my = shape[0], shape[1]
        mx, my = torch.meshgrid(torch.linspace(0, 1, mx), torch.linspace(0, 1, my), indexing='ij')
        mx, my = mx.to(device), my.to(device)
        return torch.stack([mx, my], dim=-1)
    
    def get_grid_freq(self, shape):
        mx, my = shape[0], shape[1]
        mx, my = torch.meshgrid(torch.fft.fftfreq(mx, d=1), torch.fft.rfftfreq(my, d=1), indexing='ij')
        mx, my = mx.to(device), my.to(device)
        return torch.stack([mx, my], dim=-1)
    
class NS(nn.Module):
    def __init__(self, width, symbol_act, use_last=False, num_layer=2, num_hidden=32, net_out=True):
        super(NS, self).__init__()
        self.width = width
        self.use_last = use_last
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(12, self.width)
        if use_last:
            self.fc0 = nn.Linear(3, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, num_layer=num_layer, num_hidden=num_hidden, symbol_act= symbol_act, net_out=net_out)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, num_layer=num_layer, num_hidden=num_hidden, symbol_act= symbol_act, net_out=net_out)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, num_layer=num_layer, num_hidden=num_hidden, symbol_act= symbol_act, net_out=net_out)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, num_layer=num_layer, num_hidden=num_hidden, symbol_act= symbol_act, net_out=net_out)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def cal_weights(self, shape):
        for mod in list(self.children()):
            if isinstance(mod, SpectralConv2d_fast):
                mod.cal_weights(shape)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        if self.use_last:
            x = x[...,-1:]
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
def get_args(argv=None):
    parser = argparse.ArgumentParser(description = 'Put your hyperparameters')
    parser.add_argument('name', type=str, help='experiments name')
    parser.add_argument('--batch', default=20, type=int, help = 'batch size')
    parser.add_argument('--epochs', default=1000, type=int, help = 'Number of Epochs')
    parser.add_argument('--lr', default=5e-3, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-6, type=float, help='weight decay')
    parser.add_argument('--step_size', default=200, type=int, help='scheduler step size')
    parser.add_argument('--gamma', default=0.5, type=float, help='scheduler factor')
    parser.add_argument('--multgpu', action='store_true', help='whether multiple gpu or not')
    parser.add_argument('--nu', default=1e-5, type=float, help='vis in NS equation')
    parser.add_argument('--width', default=20, type=int, help='number of channel')
    parser.add_argument('--num_layer', default=2, type=int, help='number of hidden layer of implicit network')
    parser.add_argument('--num_hidden', default=32, type=int, help='dimension of hidden layer of implicit network')
    parser.add_argument('--load_path', default=None, type=str, help='path of directory to resume the training')
    parser.add_argument('--act', default='gelu', type=str, help='activation')
    parser.add_argument('--num_data', default=1000, type=int, help='number of data to use, only for which nu=1e-4')
    parser.add_argument('--net_out', action='store_false', help='use symbol network with a(x) or not')
    
    return parser.parse_args(argv)
    
    
if __name__=="__main__":
    args = get_args()
    print(args)
    NAME = args.name
    if args.load_path is None:
        PATH = 'results/{}/'.format(sys.argv[0][:-3])
        if not os.path.exists(PATH):    
            os.mkdir(PATH)
        PATH = os.path.join(PATH, NAME)
        os.mkdir(PATH)
    else:
        PATH = args.load_path
        args = torch.load(os.path.join(args.load_path, 'args.bin'))
        args.load_path = PATH
        args.name = NAME
        PATH = os.path.join(PATH, NAME)
        os.mkdir(PATH)

    shutil.copy(sys.argv[0], os.path.join(PATH, 'code.py'))
    
    if args.multgpu:
        num_gpu = torch.cuda.device_count()
    else:
        num_gpu = 1
    
    lr = args.lr
    wd = args.wd
    batch_size = args.batch
    EPOCHS = args.epochs
    step_size = args.step_size
    gamma = args.gamma
    nu = args.nu
    width = args.width
    num_layer = args.num_layer
    num_hidden = args.num_hidden
    symbol_act = args.act
    net_out = args.net_out
    
    torch.save(args, os.path.join(PATH, 'args.bin'))
    
    if nu==1e-5:
        T_in = 10
        T = 10
        ntrain = 1000
        ntest = 200
        u = torch.load('data/ns_V1e-5_N1200_T20.bin')
        
    elif nu==1e-3:
        T_in = 10
        T = 40
        ntrain = 1000
        ntest = 200
        u = torch.load('data/ns_V1e-3_N5000_T50.bin')
        
    elif nu==1e-4:
        T_in = 10
        T = 20
        ntrain = args.num_data
        ntest = 200
        u = torch.load('data/ns_V1e-4_N10000_T30.bin')
        
    x_train = u[:ntrain,:,:,:T_in]
    y_train = u[:ntrain,:,:,T_in:T+T_in]
    trainset = torch.utils.data.TensorDataset(x_train, y_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_gpu)
    x_test = u[-ntest:,:,:,:T_in]
    y_test = u[-ntest:,:,:,T_in:T+T_in]
    testset = torch.utils.data.TensorDataset(x_test, y_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_gpu)
        
    model = NS(width=width, symbol_act=symbol_act, use_last=False, num_hidden=num_hidden, num_layer=num_layer, net_out=net_out).to(device)
    if num_gpu> 1:
        print("Let's use", num_gpu, "GPUs!")
        model = nn.DataParallel(model).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    start_epoch = 0
    
    if args.load_path is not None:
        model.load_state_dict(torch.load(os.path.join(args.load_path, 'weight.bin')))
        checkpoint = torch.load(os.path.join(args.load_path, 'checkpoint.bin'))
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        print("Load previous checkpoints from {}".format(args.load_path))
        print("Resume from %d epoch (reamining %d epochs)"%(start_epoch, EPOCHS-start_epoch))
    
    train_rel = []
    test_rel = []
    pbar = tqdm(total=EPOCHS-start_epoch, file=sys.stdout)
    for epoch in range(1+start_epoch, EPOCHS+1):
        model.train()
        train_l2_full = 0
        for u_prev, u in trainloader:
            optimizer.zero_grad()
            #model.cal_weights(u_prev.shape[1:3])
            loss = 0
            u_prev, u = u_prev.to(device), u.to(device)
            for t in range(T):                
                u_target = u[..., t:t+1]
                u_pred = model(u_prev)
                loss = loss+rel_error(u_pred, u_target).sum()
                if t==0:
                    pred = u_pred
                else:
                    pred = torch.cat([pred, u_pred], dim=-1)
                u_prev = torch.cat([u_prev[..., 1:], u_pred], dim=-1)                
            loss.backward()
            optimizer.step()
                
            l2_full = rel_error(pred, u).sum()
            train_l2_full += l2_full.item()
        train_rel.append(train_l2_full/ntrain)
        
        model.eval()
        #model.cal_weights(u_prev.shape[1:3])
        with torch.no_grad():
            test_l2_full=0     
            for u_prev, u in testloader:
                u_prev, u = u_prev.to(device), u.to(device)
                for t in range(T):                
                    u_target = u[..., t:t+1]
                    u_pred = model(u_prev)
                    if t==0:
                        pred = u_pred
                    else:
                        pred = torch.cat([pred, u_pred], dim=-1)
                    u_prev = torch.cat([u_prev[..., 1:], u_pred], dim=-1)
                l2_full = rel_error(pred, u).sum()
                test_l2_full += l2_full.item()
            test_rel.append(test_l2_full/ntest)
        pbar.set_description("###### Epoch : %d, Loss_train : %.4f, Loss_test : %.4f ######"%(epoch, train_rel[-1], test_rel[-1]))
        scheduler.step()
        pbar.update()
        torch.save(model.state_dict(),os.path.join(PATH, 'weight.bin'))
        torch.save({'train_rel':train_rel, 'test_rel':test_rel}, os.path.join(PATH, 'loss.bin'))
        torch.save({'epoch':epoch, 
                    'optimizer':optimizer.state_dict(),
                    'scheduler':scheduler.state_dict()}, os.path.join(PATH, 'checkpoint.bin'))

