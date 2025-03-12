'''Helper functions.'''
import numpy as np
import torch
from scipy.linalg import sqrtm, fractional_matrix_power

def float_x(data):
    '''Set data array precision.'''
    return np.float32(data)

def matrix_power(M, power):
    if len(M.shape) == 2:
        assert M.shape[0] == M.shape[1], "Matrix must be square"
        try:
            # gpu square root
            S, U = torch.linalg.eigh(M)
            S[S<0] = 0.
            return U @ torch.diag(S**power) @ U.T
        except:
            # stable cpu square root
            M.diagonal().add_(1e-8)
            if power == 0.5:
                sqrtM = sqrtm(M.cpu())
            else:
                sqrtM = fractional_matrix_power(M.cpu(), power)
            sqrtM = torch.from_numpy(sqrtM).to(M.device)
            return sqrtM
    elif len(M.shape) == 1:
        assert M.shape[0] > 0, "Vector must be non-empty"
        M[M<0] = 0.
        return M**power
    else:
        raise ValueError(f"Invalid matrix shape for square root: {M.shape}")