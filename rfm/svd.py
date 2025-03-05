'''Utility functions for performing fast SVD.'''
import torch
import torch.linalg as linalg
import time
from math import sqrt

def nystrom_kernel_svd(samples, kernel_fn, top_q, method='lobpcg'):
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

    # vals, vecs = linalg.eigh(scaled_kmat.cuda())
    if method == 'lobpcg':
        start_time = time.time()
        def custom_tracker(lobpcg_instance):
            current_step = lobpcg_instance.ivars["istep"]
            converged_count = lobpcg_instance.ivars["converged_count"]
            # current_eigenvalues = lobpcg_instance.E[:lobpcg_instance.iparams["k"]]
            current_residual = lobpcg_instance.tvars["rerr"]
            elapsed_time = time.time() - start_time
            print(f"LOBPCG Step {current_step}: Converged {converged_count} eigenpairs, Time: {elapsed_time:.2f}s")
            # print(f"Current residual: {current_residual[-100:]}")

        print(f"Using LOBPCG with n_samples={n_samples}, top_q={top_q}")
        X = torch.randn(scaled_kmat.shape[0], top_q, device='cuda')
        vals, vecs = torch.lobpcg(scaled_kmat.cuda(), method='ortho', k=top_q, X=X, largest=True, tracker=custom_tracker, niter=50)
        vals = torch.flip(vals, dims=(0,))
        vecs = torch.flip(vecs, dims=(1,))
    elif method == 'eigh':
        vals, vecs = linalg.eigh(scaled_kmat.cuda())

    vals = vals.float()
    vecs = vecs.float()
    eigvals = torch.flip(vals, dims=(0,))[:top_q]
    eigvecs = torch.flip(vecs, dims=(1,))[:, :top_q] / sqrt(n_samples)
    return eigvals.float(), eigvecs.float()
