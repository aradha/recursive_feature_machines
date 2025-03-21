import numpy as np
import torch
from rfm import LaplaceRFM, GeneralizedLaplaceRFM
from rfm.generic_kernels import LaplaceKernel, ProductLaplaceKernel, LpqLaplaceKernel
from rfm.generic_kernels import SumPowerLaplaceKernel
from rfm.recursive_feature_machine import GenericRFM
import time

np.random.seed(0)
torch.manual_seed(0)

M_batch_size = 256

# def fstar(X):
#     return torch.cat([
#             (X[:, 0]  > 0)[:,None],
#     	    (X[:, 1] < 0.5)[:, None]], 
#     	    axis=1
#         ).float()

def fstar(X):
    return (X[:, 0] ** 2)[:,None].float()

n = 500 # samples
d = 100  # dimension

bw = 200.
reg = 1e-8
iters=3

X_train = torch.randn(n, d).cuda()
X_test = torch.randn(n, d).cuda()
y_train = fstar(X_train).cuda()
y_test = fstar(X_test).cuda()

print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')

model = LaplaceRFM(bandwidth=bw, diag=False, reg=reg, device='cuda', centering=True)


start_time = time.time()
model.fit(
    (X_train, y_train), 
    (X_test, y_test), 
    iters=iters,
    classification=False,
    M_batch_size=len(X_train),
)

print(f'LaplaceRFM Time: {time.time()-start_time:g} s')


model = GenericRFM(LaplaceKernel(bandwidth=bw, exponent=1.0), diag=False, reg=reg, device='cuda', centering=True)

start_time = time.time()

model.fit(
    (X_train, y_train), 
    (X_test, y_test), 
    iters=iters,
    classification=False,
    M_batch_size=len(X_train),
)

print(f'Generic time: {time.time()-start_time:g} s')

