'''Helper functions.'''
import numpy as np
import torch
from scipy.linalg import sqrtm, fractional_matrix_power

def float_x(data):
    '''Set data array precision.'''
    return np.float32(data)

def matrix_power(M, power):
    """
    Compute the power of a matrix.
    :param M: Matrix to power.
    :param power: Power to raise the matrix to.
    :return: Matrix raised to the power - M^{power}.
    """
    if len(M.shape) == 2:
        assert M.shape[0] == M.shape[1], "Matrix must be square"

        # gpu square root
        S, U = torch.linalg.eigh(M)
        S[S<0] = 0.
        return U @ torch.diag(S**power) @ U.T
    elif len(M.shape) == 1:
        assert M.shape[0] > 0, "Vector must be non-empty"
        M[M<0] = 0.
        return M**power
    else:
        raise ValueError(f"Invalid matrix shape for square root: {M.shape}")
    
def get_data_from_loader(data_loader):
    """
    Get data from a data loader.
    :param data_loader: Torch DataLoader to get data from.
    :return: Tuple of tensors - (X, y).
    """
    X, y = [], []
    for idx, batch in enumerate(data_loader):
        inputs, labels = batch
        X.append(inputs)
        y.append(labels)
    return torch.cat(X, dim=0), torch.cat(y, dim=0)