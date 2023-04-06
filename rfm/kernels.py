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
