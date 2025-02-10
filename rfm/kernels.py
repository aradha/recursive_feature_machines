'''Implementation of kernel functions.'''

import torch


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
    kernel_mat = euclidean_distances_M(samples, centers, M, squared=False)
    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat

def laplacian_M_grad1(samples, centers, M, bandwidth):
    assert bandwidth > 0
    kernel_mat = laplacian_M(samples, centers, M, bandwidth)
    dist = euclidean_distances_M(samples, centers, M, squared=False)
    dist = torch.where(dist < 1e-10, torch.zeros(1).float(), dist)

    kernel_mat = kernel_mat/dist
    kernel_mat[kernel_mat == float("Inf")] = 0.
    return -kernel_mat/bandwidth


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
    kernel_mat = euclidean_distances_M(samples, centers, M, squared=True)
    kernel_mat.clamp_(min=0)
    gamma = 1. / (2 * bandwidth ** 2)
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat

def gaussian_M_grad1(samples, centers, M, bandwidth):
    assert bandwidth > 0
    kernel_mat = gaussian_M(samples, centers, M, bandwidth)
    return -kernel_mat/bandwidth**2


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






#### LAPLACIAN GEN FUNCTIONS #### 

def laplacian_gen(X: torch.Tensor, Z: torch.Tensor, sqrtM: torch.Tensor = None, L: float=10.0, exponent: float = 1.0) -> torch.Tensor:
    """
    Memory-efficient implementation of exponential kernel k(x,z) = prod_{i=1}^d exp(-|xi-zi|^v)
    using torch.vmap for vectorization.
    
    Args:
        X: Input tensor of shape (n, d)
        Z: Input tensor of shape (m, d)
        v: Power parameter for the kernel (default: 1.0)
    
    Returns:
        Kernel matrix of shape (n, m)
    """
    n, d = X.shape
    m, d2 = Z.shape
    assert d == d2, "Feature dimensions must match"

    if sqrtM is not None:
        X = X @ sqrtM
        Z = Z @ sqrtM
    
    def single_dim_contribution(x_i: torch.Tensor, z_i: torch.Tensor) -> torch.Tensor:
        """Compute contribution of a single dimension to the kernel."""
        # x_i shape: (n,), z_i shape: (m,)
        diff = torch.abs(x_i.unsqueeze(-1) - z_i)  / L # (n, m)
        return -torch.pow(diff, exponent)
    
    # Vectorize over the feature dimension
    vmapped_contrib = torch.vmap(single_dim_contribution, in_dims=1)
    
    # Compute all contributions and sum in log space
    log_kernel = vmapped_contrib(X, Z).sum(dim=0)  # sum over feature dimension
    
    return torch.exp(log_kernel)


def get_laplacian_gen_grad(
    x: torch.Tensor, 
    z: torch.Tensor, 
    sqrtM: torch.Tensor, 
    v: float, 
    L: float,
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
    
    # Transform x through linear layer
    Mx = x @ sqrtM
    z = z @ sqrtM
    
    # Compute pairwise differences
    diff = Mx.unsqueeze(1) - z.unsqueeze(0)  # (n, m, d_out)
    abs_diff = torch.abs(diff) / L
    
    # Compute kernel components
    sum_pow = (abs_diff ** v).sum(dim=-1)  # (n, m)
    k = torch.exp(-sum_pow)  # (n, m)
    
    # Compute gradient components for ∂k/∂(Mx)
    sign = torch.sign(diff)
    zero_mask = (abs_diff < eps)
    safe_abs = torch.where(zero_mask, torch.tensor(eps, device=x.device), abs_diff)
    dk_dMx = -v * sign * (safe_abs ** (v-1)) * k.unsqueeze(-1)  # (n, m, d_out)
    dk_dMx = torch.where(zero_mask, torch.zeros_like(dk_dMx), dk_dMx)
    
    # Backprop through linear layer: ∂k/∂x = ∂k/∂(Mx) @ M
    dk_dx = torch.einsum('nmo,do->nmd', dk_dMx, sqrtM)  # (n, m, d_in)
    
    return dk_dx.squeeze()

def get_laplace_gen_agop(
    x: torch.Tensor, 
    z: torch.Tensor, 
    sqrtM: torch.Tensor, 
    v: float, 
    L: float,
    alphas: torch.Tensor,
) -> torch.Tensor:

    dk_dx = get_laplacian_gen_grad(x, z, sqrtM, v, L)
    grads = torch.einsum('nmd,mc->ncd', dk_dx, alphas)
    grads = grads.reshape(-1, grads.shape[-1])
    agop = grads.T@grads
    return agop
