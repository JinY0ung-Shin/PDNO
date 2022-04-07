import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple

import numpy as np
import math
import os
import h5py

from functools import partial

from .models.utils import train, test, LpLoss, get_filter, UnitGaussianNormalizer
import argparse
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_initializer(name):
    
    if name == 'xavier_normal':
        init_ = partial(nn.init.xavier_normal_)
    elif name == 'kaiming_uniform':
        init_ = partial(nn.init.kaiming_uniform_)
    elif name == 'kaiming_normal':
        init_ = partial(nn.init.kaiming_normal_)
    return init_

class sparseKernel(nn.Module):
    def __init__(self,
                 k, alpha, c=1, 
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernel,self).__init__()
        
        self.k = k
        self.conv = self.convBlock(alpha*k**2, alpha*k**2)
        self.Lo = nn.Conv1d(alpha*k**2, c*k**2, 1)
        
    def forward(self, x):
        B, c, ich, Nx, Ny, T = x.shape # (B, c, ich, Nx, Ny, T)
        x = x.reshape(B, -1, Nx, Ny, T)
        x = self.conv(x)
        x = self.Lo(x.view(B, c*ich, -1)).view(B, c, ich, Nx, Ny, T)
        return x
        
        
    def convBlock(self, ich, och):
        net = nn.Sequential(
            nn.Conv3d(och, och, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        return net 
    

def compl_mul3d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    op = partial(torch.einsum, "bixyz,ioxyz->boxyz")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)


# fft conv taken from: https://github.com/zongyi-li/fourier_neural_operator
class sparseKernelFT(nn.Module):
    def __init__(self,
                 k, alpha, c=1, 
                 nl = 1,
                 initializer = None,
                 **kwargs):
        super(sparseKernelFT, self).__init__()        
        
        self.modes = alpha

        self.weights1 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes, self.modes, self.modes, 2))
        self.weights2 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes, self.modes, self.modes, 2))        
        self.weights3 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes, self.modes, self.modes, 2))        
        self.weights4 = nn.Parameter(torch.zeros(c*k**2, c*k**2, self.modes, self.modes, self.modes, 2))        
        nn.init.xavier_normal_(self.weights1)
        nn.init.xavier_normal_(self.weights2)
        nn.init.xavier_normal_(self.weights3)
        nn.init.xavier_normal_(self.weights4)
        
        self.Lo = nn.Conv1d(c*k**2, c*k**2, 1)
#         self.Wo = nn.Conv1d(c*k**2, c*k**2, 1)
        self.k = k
        
    def forward(self, x):
        B, c, ich, Nx, Ny, T = x.shape # (B, c, ich, N, N, T)
        
        x = x.reshape(B, -1, Nx, Ny, T)
        x_fft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        x_fft = torch.stack([x_fft.real, x_fft.imag], dim=-1)
        # Multiply relevant Fourier modes
        l1 = min(self.modes, Nx//2+1)
        l2 = min(self.modes, Ny//2+1)
        l3 = min(self.modes, T//2+1)
        out_ft = torch.zeros(B, c*ich, Nx, Ny, T//2 +1, 2, device=x.device)
        
        
        out_ft[:, :, :l1, :l2, :l3] = compl_mul3d(
            x_fft[:, :, :l1, :l2, :l3], self.weights1[:, :, :l1, :l2, :l3])
        out_ft[:, :, -l1:, :l2, :l3] = compl_mul3d(
                x_fft[:, :, -l1:, :l2, :l3], self.weights2[:, :, :l1, :l2, :l3])
        out_ft[:, :, :l1, -l2:, :l3] = compl_mul3d(
                x_fft[:, :, :l1, -l2:, :l3], self.weights3[:, :, :l1, :l2, :l3])
        out_ft[:, :, -l1:, -l2:, :l3] = compl_mul3d(
                x_fft[:, :, -l1:, -l2:, :l3], self.weights4[:, :, :l1, :l2, :l3])
        
        #Return to physical space
        out_ft = torch.complex(out_ft[...,0], out_ft[..., 1])
        x = torch.fft.irfftn(out_ft, s= (Nx, Ny, T))
        
        x = F.relu(x)
        x = self.Lo(x.view(B, c*ich, -1)).view(B, c, ich, Nx, Ny, T)
        return x
        
    
class MWT_CZ(nn.Module):
    def __init__(self,
                 k = 3, alpha = 5, 
                 L = 0, c = 1,
                 base = 'legendre',
                 initializer = None,
                 **kwargs):
        super(MWT_CZ, self).__init__()
        
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0@PHI0
        G0r = G0@PHI0
        H1r = H1@PHI1
        G1r = G1@PHI1
        
        H0r[np.abs(H0r)<1e-8]=0
        H1r[np.abs(H1r)<1e-8]=0
        G0r[np.abs(G0r)<1e-8]=0
        G1r[np.abs(G1r)<1e-8]=0
        
        self.A = sparseKernelFT(k, alpha, c)
        self.B = sparseKernelFT(k, alpha, c)
        self.C = sparseKernelFT(k, alpha, c)
        
        self.T0 = nn.Conv1d(c*k**2, c*k**2, 1)

        if initializer is not None:
            self.reset_parameters(initializer)

        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((np.kron(H0, H0).T, 
                            np.kron(H0, H1).T,
                            np.kron(H1, H0).T,
                            np.kron(H1, H1).T,
                           ), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((np.kron(G0, G0).T,
                            np.kron(G0, G1).T,
                            np.kron(G1, G0).T,
                            np.kron(G1, G1).T,
                           ), axis=0)))
        
        self.register_buffer('rc_ee', torch.Tensor(
            np.concatenate((np.kron(H0r, H0r), 
                            np.kron(G0r, G0r),
                           ), axis=0)))
        self.register_buffer('rc_eo', torch.Tensor(
            np.concatenate((np.kron(H0r, H1r), 
                            np.kron(G0r, G1r),
                           ), axis=0)))
        self.register_buffer('rc_oe', torch.Tensor(
            np.concatenate((np.kron(H1r, H0r), 
                            np.kron(G1r, G0r),
                           ), axis=0)))
        self.register_buffer('rc_oo', torch.Tensor(
            np.concatenate((np.kron(H1r, H1r), 
                            np.kron(G1r, G1r),
                           ), axis=0)))
        
        
    def forward(self, x):
        
        B, c, ich, Nx, Ny, T = x.shape # (B, c, k^2, Nx, Ny, T)
        ns = math.floor(np.log2(Nx))

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

#         decompose
        for i in range(ns-self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x.reshape(B, c*ich, -1)).view(
            B, c, ich, 2**self.L, 2**self.L, T) # coarsest scale transform

#        reconstruct            
        for i in range(ns-1-self.L,-1,-1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), 2)
            x = self.evenOdd(x)

        return x

    
    def wavelet_transform(self, x):
        xa = torch.cat([x[:, :, :, ::2 , ::2 , :], 
                        x[:, :, :, ::2 , 1::2, :], 
                        x[:, :, :, 1::2, ::2 , :], 
                        x[:, :, :, 1::2, 1::2, :]
                       ], 2)
        waveFil = partial(torch.einsum, 'bcixyt,io->bcoxyt') 
        d = waveFil(xa, self.ec_d)
        s = waveFil(xa, self.ec_s)
        return d, s
        
        
    def evenOdd(self, x):
        
        B, c, ich, Nx, Ny, T = x.shape # (B, c, 2*k^2, Nx, Ny)
        assert ich == 2*self.k**2
        evOd = partial(torch.einsum, 'bcixyt,io->bcoxyt')
        x_ee = evOd(x, self.rc_ee)
        x_eo = evOd(x, self.rc_eo)
        x_oe = evOd(x, self.rc_oe)
        x_oo = evOd(x, self.rc_oo)
        
        x = torch.zeros(B, c, self.k**2, Nx*2, Ny*2, T,
            device = x.device)
        x[:, :, :, ::2 , ::2 , :] = x_ee
        x[:, :, :, ::2 , 1::2, :] = x_eo
        x[:, :, :, 1::2, ::2 , :] = x_oe
        x[:, :, :, 1::2, 1::2, :] = x_oo
        return x
    
    def reset_parameters(self, initializer):
        initializer(self.T0.weight)
    
    
class MWT(nn.Module):
    def __init__(self,
                 ich = 1, k = 3, alpha = 2, c = 1,
                 nCZ = 3,
                 L = 0,
                 base = 'legendre',
                 initializer = None,
                 **kwargs):
        super(MWT,self).__init__()
        
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ
        self.Lk = nn.Linear(ich, c*k**2)
        
        self.MWT_CZ = nn.ModuleList(
            [MWT_CZ(k, alpha, L, c, base, 
            initializer) for _ in range(nCZ)]
        )
        self.BN = nn.ModuleList(
            [nn.BatchNorm3d(c*k**2) for _ in range(nCZ)]
        )
        self.Lc0 = nn.Linear(c*k**2, 128)
        self.Lc1 = nn.Linear(128, 1)
        
        if initializer is not None:
            self.reset_parameters(initializer)
        
    def forward(self, x):
        
        B, Nx, Ny, T, ich = x.shape # (B, Nx, Ny, T, d)
        ns = math.floor(np.log2(Nx))
        x = self.Lk(x)
        x = x.view(B, Nx, Ny, T, self.c, self.k**2)
        x = x.permute(0, 4, 5, 1, 2, 3)
    
        for i in range(self.nCZ):
            x = self.MWT_CZ[i](x)
            x = self.BN[i](x.view(B, -1, Nx, Ny, T)).view(
                B, self.c, self.k**2, Nx, Ny, T)
            if i < self.nCZ-1:
                x = F.relu(x)

        x = x.view(B, -1, Nx, Ny, T) # collapse c and k**2
        x = x.permute(0, 2, 3, 4, 1)
        x = self.Lc0(x)
        x = F.relu(x)
        x = self.Lc1(x)
        return x.squeeze()
    
    def reset_parameters(self, initializer):
        initializer(self.Lc0.weight)
        initializer(self.Lc1.weight)
        
def get_args(argv=None):
    parser = argparse.ArgumentParser(description = 'Put your hyperparameters')
    parser.add_argument('name', type=str, help='experiments name')
    parser.add_argument('--batch', default=10, type=int, help = 'batch size')
    parser.add_argument('--epochs', default=500, type=int, help = 'Number of Epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--step_size', default=100, type=int, help='scheduler step size')
    parser.add_argument('--gamma', default=0.5, type=float, help='scheduler factor')
    parser.add_argument('--multgpu', action='store_true', help='whether multiple gpu or not')
    parser.add_argument('--nu', default=1e-5, type=float, help='vorticity in NS equation')
    parser.add_argument('--num_data', default=1000, type=int, help='number of data to use, only for which nu=1e-4')
    
    return parser.parse_args(argv)
        
if __name__=='__main__':
    args = get_args()
    print(args)    
    
    nu = args.nu
    S=64
    sub=1
    if nu==1e-5:
        T_in = 10
        T = 10
        ntrain = 1000
        ntest = 200
        u = torch.load('../data/ns_V1e-5_N1200_T20.bin')
    elif nu==1e-3:
        T_in = 10
        T = 40
        ntrain = 1000
        ntest = 200
        u = torch.load('../data/ns_V1e-3_N5000_T50.bin')
    elif nu==1e-4:
        T_in = 10
        T = 20
        ntrain = args.num_data
        ntest = 200
        u = torch.load('../data/ns_V1e-4_N10000_T30.bin')
    train_a = u[:ntrain,:,:,:T_in]
    train_u = u[:ntrain,:,:,T_in:T+T_in]
    test_a= u[-ntest:,:,:,:T_in]
    test_u = u[-ntest:,:,:,T_in:T+T_in]
    
    a_normalizer = UnitGaussianNormalizer(train_a)
    x_train = a_normalizer.encode(train_a)
    x_test = a_normalizer.encode(test_a)

    y_normalizer = UnitGaussianNormalizer(train_u)
    y_train = y_normalizer.encode(train_u)

    x_train = x_train.reshape(ntrain,S,S,1,T_in).repeat([1,1,1,T,1])
    x_test = x_test.reshape(ntest,S,S,1,T_in).repeat([1,1,1,T,1])
    
    # pad locations (x,y,t)
    gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
    gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
    gridt = torch.tensor(np.linspace(0, 1, T+1)[1:], dtype=torch.float)
    gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])

    x_train = torch.cat((gridx.repeat([ntrain,1,1,1,1]), gridy.repeat([ntrain,1,1,1,1]),
                           gridt.repeat([ntrain,1,1,1,1]), x_train), dim=-1)
    x_test = torch.cat((gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1]),
                           gridt.repeat([ntest,1,1,1,1]), x_test), dim=-1)

    batch_size = args.batch
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, test_u), batch_size=batch_size, shuffle=False)
    
    ich = 13
    initializer = get_initializer('xavier_normal') # xavier_normal, kaiming_normal, kaiming_uniform

    torch.manual_seed(0)
    np.random.seed(0)

    alpha = 12
    c = 4
    k = 3
    nCZ = 4
    L = 0
    model = MWT(ich, 
                alpha = alpha,
                c = c,
                k = k, 
                base = 'legendre', # chebyshev
                nCZ = nCZ,
                L = L,
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
    torch.save({'train_rel':train_loss, 'test_rel':test_loss}, 'results/loss_{}.bin'.format(args.name))
    torch.save(model.state_dict(), 'results/weight_{}.bin'.format(args.name))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
