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

# model = LaplaceRFM(bandwidth=10., diag=False, reg=1e-4, device='cuda')
model = GenericRFM(LaplaceKernel(bandwidth=10., exponent=1.0), reg=1e-4, device='cuda')
# model = GenericRFM(LpqLaplaceKernel(bandwidth=50., p=1.5, q=0.7), diag=True, reg=1e-4)
# model = GeneralizedLaplaceRFM(bandwidth=50., exponent=1.0, diag=True)
# model = GenericRFM(SumPowerLaplaceKernel(bandwidth=10., exponent=1.0, power=5, const_mix=0), diag=True, reg=1e-4)

n = 500 # samples
d = 4000  # dimension

iters=3

X_train = torch.randn(n, d).cuda()
X_test = torch.randn(n, d).cuda()
y_train = fstar(X_train).cuda()
y_test = fstar(X_test).cuda()

print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')

start_time = time.time()
model.fit(
    (X_train, y_train), 
    (X_test, y_test), 
    iters=iters,
    classification=False,
    M_batch_size=len(X_train),
)

print(f'LaplaceRFM Time: {time.time()-start_time:g} s')


model = GenericRFM(LaplaceKernel(bandwidth=10., exponent=1.0), diag=False, reg=1e-4, device='cuda')

start_time = time.time()

model.fit(
    (X_train, y_train), 
    (X_test, y_test), 
    iters=iters,
    classification=False,
    M_batch_size=len(X_train),
)

print(f'Generic time: {time.time()-start_time:g} s')

# M, err = rfm((X_train, y_train), (X_test, y_test), num_iters=iters)
# print(f'RFM time: {time.time()-start_time:g} s')
# print(f'RFM err: {err}')
