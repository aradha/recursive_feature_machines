'''Implementation of kernel functions.'''

import torch
import numpy as np
from typing import Union

def euclidean_distances(samples, centers, squared=True):
    samples_norm2 = samples.pow(2).sum(-1)
    if samples is centers:
        centers_norm2 = samples_norm2
    else:
        centers_norm2 = centers.pow(2).sum(-1)

    distances = -2 * samples @ centers.T
    distances.add_(samples_norm2.view(-1, 1))
    distances.add_(centers_norm2)
    if not squared:
        distances.clamp_(min=0).sqrt_()

    return distances

def euclidean_distances_M(samples, centers, M, squared=True):
    if len(M.shape)==1:
        return euclidean_distances_M_diag(samples, centers, M, squared=squared)

    samples_norm2 = ((samples @ M) * samples).sum(-1)

    if samples is centers:
        centers_norm2 = samples_norm2
    else:
        centers_norm2 = ((centers @ M) * centers).sum(-1)

    distances = -2 * (samples @ M) @ centers.T
    distances.add_(samples_norm2.view(-1, 1))
    distances.add_(centers_norm2)

    if not squared:
        distances.clamp_(min=0).sqrt_()

    return distances

def euclidean_distances_M_diag(samples, centers, M, squared=True):
    "assumes M is a diagonal matrix"
    samples_norm2 = ((samples * M) * samples).sum(-1)

    if samples is centers:
        centers_norm2 = samples_norm2
    else:
        centers_norm2 = ((centers * M) * centers).sum(-1)

    distances = -2 * (samples * M) @ centers.T
    distances.add_(samples_norm2.view(-1, 1))
    distances.add_(centers_norm2)

    if not squared:
        distances.clamp_(min=0).sqrt_()

    return distances

def laplacian(samples, centers, bandwidth):
    '''Laplacian kernel.

    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        bandwidth: kernel bandwidth.

    Returns:
        kernel matrix of shape (n_sample, n_center).
    '''
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers, squared=False)
    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat

def laplacian_M(samples, centers, M, bandwidth):
    assert bandwidth > 0
    if M is None:
        kernel_mat = euclidean_distances(samples, centers, squared=False)
    else:
        kernel_mat = euclidean_distances_M(samples, centers, M, squared=False)
    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat

def gaussian(samples, centers, bandwidth):
    '''Gaussian kernel.

    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        bandwidth: kernel bandwidth.

    Returns:
        kernel matrix of shape (n_sample, n_center).
    '''
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers, squared=True)
    kernel_mat.clamp_(min=0)
    gamma = 1. / (2 * bandwidth ** 2)
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat


def gaussian_M(samples, centers, M, bandwidth):
    assert bandwidth > 0
    if M is None:
        kernel_mat = euclidean_distances(samples, centers, squared=True)
    else:
        kernel_mat = euclidean_distances_M(samples, centers, M, squared=True)
    kernel_mat.clamp_(min=0)
    gamma = 1. / (2 * bandwidth ** 2)
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat


def dispersal(samples, centers, bandwidth, gamma):
    '''Dispersal kernel.

    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        bandwidth: kernel bandwidth.
        gamma: dispersal factor.

    Returns:
        kernel matrix of shape (n_sample, n_center).
    '''
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers)
    kernel_mat.pow_(gamma / 2.)
    kernel_mat.mul_(-1. / bandwidth)
    kernel_mat.exp_()
    return kernel_mat


#### NTK FUNCTIONS ####

def ntk_kernel(pair1, pair2):

    out = pair1 @ pair2.transpose(1, 0) + 1
    N1 = torch.sum(torch.pow(pair1, 2), dim=-1).view(-1, 1) + 1
    N2 = torch.sum(torch.pow(pair2, 2), dim=-1).view(-1, 1) + 1

    XX = torch.sqrt(N1 @ N2.transpose(1, 0))
    out = out / XX

    out = torch.clamp(out, -1, 1)

    first = 1/np.pi * (out * (np.pi - torch.acos(out)) \
                       + torch.sqrt(1. - torch.pow(out, 2))) * XX
    sec = 1/np.pi * out * (np.pi - torch.acos(out)) * XX
    out = first + sec

    # Set C below as small as possible for fast convergence
    # C = 1 on real data usually works well
    # set C > 1 if EigenPro is not converging
    C = 1
    return out / C


#### LAPLACIAN GEN FUNCTIONS #### 

def laplacian_gen(X: torch.Tensor, Z: torch.Tensor, sqrtM: torch.Tensor = None, L: float = 10.0, v: float = 1.0, diag: bool = False) -> torch.Tensor:
    """
    Optimized memory-efficient implementation of exponential kernel using batched tensor operations.
    
    Args:
        X: Input tensor of shape (n, d)
        Z: Input tensor of shape (m, d)
        sqrtM: Optional transformation matrix
        L: Length scale parameter (default: 10.0)
        exponent: Power parameter for the kernel (default: 1.0)
        batch_size: Number of dimensions to process at once (default: 50)
    
    Returns:
        Kernel matrix of shape (n, m)
    """
    n, d = X.shape
    m, d2 = Z.shape
    assert d == d2, "Feature dimensions must match"

    if sqrtM is not None:
        if diag:
            assert sqrtM.shape == (d,), "sqrtM must be a vector of length d"
            X = X * sqrtM.view(1, -1)
            Z = Z * sqrtM.view(1, -1)
        else:
            X = X @ sqrtM
            Z = Z @ sqrtM

    pdists = torch.cdist(X/L, Z/L, p=v) ## ||X/L-Z/L||_p = (\sum_{i=1}^d (|xi-zi|/L)^p)^(1/p)
    pdists_p = pdists**v ## \sum_{i=1}^d (|xi-zi|/L)^p
    return torch.exp(-1*pdists_p) ## \prod_{i=1}^d exp(-(|xi-zi|/L)^p)

def get_laplacian_gen_grad(
    x: torch.Tensor, 
    z: torch.Tensor, 
    sqrtM: Union[torch.Tensor, None], 
    v: float, 
    L: float,
    alphas: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Computes dk/dx for the kernel k(Mx, z) = ∏ exp(-|(Mx)_i - z_i|^v)
    
    Args:
        x: Input tensor (n, d_in) or (d_in,)
        z: Input tensor (m, d_out) or (d_out,)
        sqrtM: Transformation matrix (d_in, d_out)
        v: Exponent parameter
        L: bandwidth
        eps: Numerical stability term
        
    Returns:
        Gradient tensor of shape:
        - (n, m, d_in) if x is 2D and z is 2D
        - (m, d_in) if x is 1D and z is 2D
        - (d_in,) if both are 1D
    """
    # Ensure 2D tensors
    x = x.unsqueeze(0) if x.dim() == 1 else x
    z = z.unsqueeze(0) if z.dim() == 1 else z
    
    if sqrtM is None:
        sqrtM = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    
    # Transform x through linear layer
    Mx = x @ sqrtM / L
    z = z @ sqrtM / L
    
    pdists = torch.cdist(Mx, z, p=v) ## ||X/L-Z/L||_p = (\sum_{i=1}^d (|xi-zi|/L)^p)^(1/p)
    pdists_p = pdists**v ## \sum_{i=1}^d (|xi-zi|/L)^p
    k = torch.exp(-1*pdists_p) ## \prod_{i=1}^d exp(-(|xi-zi|/L)^p)
    
    # Compute gradient components for ∂k/∂(Mx)
    zero_mask = (pdists < eps) # (n, m)
    
    diff = x.unsqueeze(1) - z.unsqueeze(0)
    zero_mask_expanded = zero_mask.unsqueeze(-1)
    safe_abs = torch.where(zero_mask_expanded, torch.tensor(eps, device=Mx.device), torch.abs(diff))
    dk_dMx = -v * torch.sign(diff) * (safe_abs ** (v-1)) * k.unsqueeze(-1)
    dk_dMx = torch.where(zero_mask_expanded, torch.zeros_like(dk_dMx), dk_dMx)
    

    # Backprop through linear layer: ∂k/∂x = ∂k/∂(Mx) @ M
    dk_dx = dk_dMx@sqrtM  # (batch, m, d_in)
    print(f'{dk_dx.transpose(1,-1).shape=}, {alphas.shape=}')
    dk_dx_sum = dk_dx.transpose(1,-1)@alphas # nmd -> ndm, mc -> ndc
    return dk_dx_sum.transpose(1,-1) # ndc -> ncd    

def get_laplacian_gen_squared_grads(
    x: torch.Tensor, 
    z: torch.Tensor, 
    sqrtM: Union[torch.Tensor, None], 
    v: float, 
    L: float,
    alphas: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Computes dk/dx for the kernel k(Mx, z) = ∏ exp(-|(Mx)_i - z_i|^v)
    
    Args:
        x: Input tensor (n, d_in) or (d_in,)
        z: Input tensor (m, d_in) or (d_in,)
        sqrtM: Transformation vector (d_in,)
        v: Exponent parameter
        L: bandwidth
        eps: Numerical stability term
        
    Returns:
        Gradient tensor of shape:
        - (n, m, d_in) if x is 2D and z is 2D
        - (m, d_in) if x is 1D and z is 2D
        - (d_in,) if both are 1D
    """
    # Ensure 2D tensors
    x = x.unsqueeze(0) if x.dim() == 1 else x
    z = z.unsqueeze(0) if z.dim() == 1 else z

    n, d = x.shape
    m, d2 = z.shape
    assert d == d2, "Feature dimensions must match"
    
    if sqrtM is None:
        sqrtM = torch.ones(x.shape[1], device=x.device, dtype=x.dtype)
    sqrtM = sqrtM.view(1, -1)

    # Transform x through linear layer
    Mx = x * sqrtM / L
    z = z * sqrtM / L
    
    pdists = torch.cdist(Mx, z, p=v) ## ||X/L-Z/L||_p = (\sum_{i=1}^d (|xi-zi|/L)^p)^(1/p)
    pdists_p = pdists**v ## \sum_{i=1}^d (|xi-zi|/L)^p
    k = torch.exp(-1*pdists_p) ## \prod_{i=1}^d exp(-(|xi-zi|/L)^p)
    
    # Compute gradient components for ∂k/∂(Mx)
    zero_mask = (pdists < eps) # (n, m)
    
    diff = Mx.unsqueeze(1) - z.unsqueeze(0)
    zero_mask_expanded = zero_mask.unsqueeze(-1)
    safe_abs = torch.where(zero_mask_expanded, torch.tensor(eps, device=Mx.device), torch.abs(diff))
    dk_dMx = -v * torch.sign(diff) * (safe_abs ** (v-1)) * k.unsqueeze(-1)
    dk_dMx = torch.where(zero_mask_expanded, torch.zeros_like(dk_dMx), dk_dMx)

    # Backprop through linear layer: ∂k/∂x = ∂k/∂(Mx) @ M
    dk_dx = dk_dMx*sqrtM  # (batch, m, d_in)
    dk_dx_sum = dk_dx.transpose(1,-1)@alphas # nmd -> ndm, mc -> ndc
    dk_dx_sum = dk_dx_sum.transpose(1,-1) # ndc -> ncd 
    dk_dx_sum = dk_dx_sum.reshape(-1, d) # ncd -> (nc)d
    return (dk_dx_sum**2).sum(dim=0)

def get_laplace_gen_agop(
    x: torch.Tensor, 
    z: torch.Tensor, 
    sqrtM: Union[torch.Tensor, None], 
    L: float,
    v: float, 
    alphas: torch.Tensor,
    diag: bool = False
) -> torch.Tensor:
    

    if diag:
        squared_grads = get_laplacian_gen_squared_grads(x, z, sqrtM, v, L, alphas)
        return squared_grads
    else:
        grads = get_laplacian_gen_grad(x, z, sqrtM, v, L, alphas)
        grads = grads.reshape(-1, grads.shape[-1])
        agop = grads.T@grads
        return agop

        
