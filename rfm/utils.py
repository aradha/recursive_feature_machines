'''Helper functions.'''
import numpy as np
import torch
from scipy.linalg import sqrtm

def float_x(data):
    '''Set data array precision.'''
    return np.float32(data)

def matrix_sqrt(M, agop_power=0.5):
    if len(M.shape) == 2:
        assert M.shape[0] == M.shape[1], "Matrix must be square"
        try:
            # gpu square root
            S, U = torch.linalg.eigh(M)
            S[S<0] = 0.
            return U @ torch.diag(S**agop_power) @ U.T
        except:
            # stable cpu square root
            M.diagonal().add_(1e-8)
            sqrtM = sqrtm(M.cpu())
            sqrtM = torch.from_numpy(sqrtM).to(M.device)
            return sqrtM
    elif len(M.shape) == 1:
        assert M.shape[0] > 0, "Vector must be non-empty"
        M[M<0] = 0.
        return M**agop_power
    else:
        raise ValueError(f"Invalid matrix shape for square root: {M.shape}")

