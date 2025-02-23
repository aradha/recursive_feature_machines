'''Helper functions.'''
import numpy as np
import torch

def float_x(data):
    '''Set data array precision.'''
    return np.float32(data)

def matrix_sqrt(M):
    S, U = torch.linalg.eigh(M)
    S[S<0] = 0.
    return U @ torch.diag(S**0.5) @ U.T
