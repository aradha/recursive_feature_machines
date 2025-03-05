'''Helper functions.'''
import numpy as np
import torch

def float_x(data):
    '''Set data array precision.'''
    return np.float32(data)

def matrix_sqrt(M, agop_power=0.5):
    if len(M.shape) == 2:
        assert M.shape[0] == M.shape[1], "Matrix must be square"
        S, U = torch.linalg.eigh(M)
        S[S<0] = 0.
        return U @ torch.diag(S**agop_power) @ U.T
    elif len(M.shape) == 1:
        assert M.shape[0] > 0, "Vector must be non-empty"
        M[M<0] = 0.
        return M**agop_power
    else:
        raise ValueError(f"Invalid matrix shape for square root: {M.shape}")
