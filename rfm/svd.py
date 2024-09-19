'''Utility functions for performing fast SVD.'''
import torch
import torch.linalg as linalg
from math import sqrt

def nystrom_kernel_svd(samples, kernel_fn, top_q):
    """Compute top eigensystem of kernel matrix using Nystrom method.

    Arguments:
        samples: data matrix of shape (n_sample, n_feature).
        kernel_fn: tensor function k(X, Y) that returns kernel matrix.
        top_q: top-q eigensystem.

    Returns:
        eigvals: top eigenvalues of shape (top_q).
        eigvecs: (rescaled) top eigenvectors of shape (n_sample, top_q).
    """

    n_samples, _ = samples.shape
    kmat = kernel_fn(samples, samples)
    scaled_kmat = kmat / n_samples
    vals, vecs = linalg.eigh(scaled_kmat)
    eigvals = torch.flip(vals, dims=(0,))[:top_q]
    eigvecs = torch.flip(vecs, dims=(1,))[:, :top_q] / sqrt(n_samples)
    return eigvals.float(), eigvecs.float()
