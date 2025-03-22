from typing import Optional, Union

import torch
from tqdm import tqdm
import time
from typing import List

import torch.nn.functional as F

def get_sub_matrix(mat: Union[torch.Tensor, None], indices: torch.Tensor) -> Union[torch.Tensor, None]:
    """
    Get a submatrix of mat based on the indices.
    """
    if mat is None:
        return None
    if len(mat.shape) == 1:
        return mat[indices]
    else:
        return mat[indices][:, indices]

class Kernel:
    def __init__(self):
        self.is_adaptive_bandwidth = True
        self.handle_categorical = False

    def _get_kernel_matrix_impl(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def _get_function_grad_impl(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def _transform_m(self, x: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Applies the given transformation matrix to x.
        :param x: Points of shape (n, d_in).
        :param mat: Matrix of shape (d_in, d_out) or vector of shape (d_in,) or None.
            A vector will be interpreted as a diagonal matrix, and None as the identity matrix.
        :return: Tensor of shape (n, d_out), where d_out=d_in in case mat is a vector or None.
        """
        if mat is not None:
            if len(mat.shape) == 1:
                # diagonal
                x = x * mat[None, :].to(dtype=x.dtype)
            elif len(mat.shape) == 2:
                x = x @ mat.to(dtype=x.dtype)
            else:
                raise ValueError(f'm_matrix should have one or two dimensions, but got shape {mat.shape}')
        return x

    def _reset_adaptive_bandwidth(self):
        self.is_adaptive_bandwidth = False
        return

    def _adapt_bandwidth(self, kernel_mat: torch.Tensor, adapt_mode='median'):
        n = kernel_mat.shape[0]
        mask = ~torch.eye(n, dtype=bool, device=kernel_mat.device)
        # Get median of off-diagonal elements only
        if adapt_mode == 'median':
            bandwidth_multiplier = torch.median(kernel_mat[mask])
        elif adapt_mode == 'mean':
            bandwidth_multiplier = torch.mean(kernel_mat[mask])
        else:
            raise ValueError(f"Invalid adapt_mode: {adapt_mode}")
        self.bandwidth = self.base_bandwidth * bandwidth_multiplier.item()
        self.is_adaptive_bandwidth = True
        return
    
    def set_categorical_indices(self, numerical_indices, categorical_indices, categorical_vectors, device='cuda'):
        print("Setting categorical indices")
        self.numerical_indices = numerical_indices.to(device)
        self.categorical_indices = [categorical_indices[i].to(device) for i in range(len(categorical_indices))]
        self.categorical_vectors = [categorical_vectors[i].to(device) for i in range(len(categorical_vectors))]
        self.handle_categorical = True
        return
    
    def get_agop(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor,
                 mat: Optional[torch.Tensor] = None, center_grads: bool = False) -> torch.Tensor:
        if self.handle_categorical:
            return self.get_agop_categorical(x, z, coefs, mat, center_grads)
        else:
            # see get_function_grads
            f_grads = self.get_function_grads(x, z, coefs, mat)
            # merge output and n_z dims
            f_grads = f_grads.reshape(-1, f_grads.shape[-1])
            if center_grads:
                f_grads = f_grads - f_grads.mean(dim=0, keepdim=True)
            return f_grads.transpose(-1, -2) @ f_grads
    
    def get_agop_categorical(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor,
                             mat: Optional[torch.Tensor] = None, center_grads: bool = False) -> torch.Tensor:
        raise NotImplementedError()
        
    def get_kernel_matrix(self, x: torch.Tensor, z: torch.Tensor,
                          mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get the kernel matrix (k(x[i, :], z[j, :]))_{i,j}
        :param x: Points of shape (n_x, d_in).
        :param z: Points of shape (n_z, d_in).
        :param mat: Matrix of shape (d_in, d_out) or vector of shape (d_in,) or None. This will be applied to x and z.
        Corresponds to sqrtM in RFM.    
        :return: The kernel matrix of shape (n_x, n_z).
        """
        if self.handle_categorical:
            return self._get_kernel_matrix_categorical_impl(x, z, mat)
        else:
            return self._get_kernel_matrix_impl(x, z, mat)
    
    def _get_kernel_matrix_categorical_impl(self, x: torch.Tensor, z: torch.Tensor,
                          mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError()

    def get_kernel_matrix_symm(self, x: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        # todo: only compute certain blocks?
        return self.get_kernel_matrix(x, x, mat)

    def get_function_grads(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor,
                           mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return the matrix of function gradients at points z.
        The function is given by f_l(\cdot) = \sum_i coefs[l, i] * k(x[i], \cdot).
        :param x: Matrix of shape (n_x, d_in)
        :param z: Matrix of shape (n_z, d_in)
        :param coefs: Vector of shape (f, n_x) where f is the number of functions
        :param mat: Matrix of shape (d_in, d_out) or vector of shape (d_in)
        :return: Should return a tensor of shape (f, n_z, d_in).
        """
        grads = self._get_function_grad_impl(self._transform_m(x, mat), self._transform_m(z, mat), coefs)
        return self._transform_m(grads, mat)

    
    def get_agop_diag(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor,
                      mat: Optional[torch.Tensor] = None, center_grads: bool = False) -> torch.Tensor:
        # see get_function_grads
        f_grads = self.get_function_grads(x, z, coefs, mat)
        # merge output and n_z dims
        f_grads = f_grads.reshape(-1, f_grads.shape[-1])
        if center_grads:
            f_grads = f_grads - f_grads.mean(dim=0, keepdim=True)
        return f_grads.square().sum(dim=-2)


class LaplaceKernel(Kernel):
    def __init__(self, bandwidth: float, exponent: float, eps: float = 1e-10, bandwidth_mode: str = 'constant'):
        super().__init__()
        assert bandwidth > 0
        assert exponent > 0
        assert eps > 0
        self.bandwidth_mode = bandwidth_mode
        self.base_bandwidth = bandwidth
        self.bandwidth = bandwidth
        self.exponent = exponent
        self.eps = eps  # this one is for numerical stability

    def _get_kernel_matrix_impl(self, x: torch.Tensor, z: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        kernel_mat = torch.cdist(self._transform_m(x, mat), self._transform_m(z, mat))
        kernel_mat.clamp_(min=0)
        if not self.is_adaptive_bandwidth:
            self._adapt_bandwidth(kernel_mat)
        if self.exponent != 1.0:
            kernel_mat.pow_(self.exponent)

        kernel_mat.mul_(-1. / (self.bandwidth ** self.exponent))
        kernel_mat.exp_()
        return kernel_mat

    def _get_function_grad_impl(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(x, z)
        dists.clamp_(min=0)

        # gradient of k(x, z) = exp(-\gamma \|x - z\|^\beta) wrt z  (where \beta = self.exponent)
        # is -\gamma k(x, z) \beta \|x - z\|^{\beta - 1} (z-x)/\|x-z\| = -\gamma \beta k(x, z) \|x - z\|^{\beta-2} (z-x)
        # therefore, setting f_l (z) = \sum_i coefs[l, i] k(x[i], z), we have
        # \grad f_l(z[j]) = \sum_i coefs[l, i] M[i, j] (z[j] - x[i]),
        # where M[i, j] = -\gamma \beta k(x[i], z[j]) \|x[i] - z[j]\|^{\beta - 2}
        gamma = 1. / self.bandwidth
        kernel_mat = dists ** self.exponent
        kernel_mat.mul_(-gamma)
        kernel_mat.exp_()

        # now compute M
        mask = dists >= self.eps
        dists.clamp_(min=self.eps)
        dists.pow_(self.exponent - 2)
        kernel_mat.mul_(dists)
        kernel_mat.mul_(mask)  # this is very important for numerical stability
        kernel_mat.mul_(-gamma * self.exponent)

        # now we want result[l, j, d] = \sum_i coefs[l, i] M[i, j] (z[j, d] - x[i, d])

        # this one uses too much memory
        # return torch.einsum('li,ij,ijd->ljd', coefs, kernel_mat, (z[None, :, :] - x[:, None, :]))

        # return (coefs @ kernel_mat)[:, :, None] * z[None, :, :] - torch.einsum('li,id,ij->ljd', coefs, x, kernel_mat)
        return torch.einsum('li,ij,jd->ljd', coefs, kernel_mat, z) - torch.einsum('li,ij,id->ljd', coefs, kernel_mat, x)

        # this one is a manual version of the two-einsum version above,
        # analogous to the old implementation but with some transposed dimensions
        # z_term = (coefs @ kernel_mat)[:, :, None] * z[None, :, :]
        # x_term = kernel_mat.t() @ (coefs.t()[:, None, :] * x[:, :, None]).reshape(x.shape[0], -1)
        # x_term = x_term.reshape(x.shape[0], x.shape[1], coefs.shape[0]).permute(2, 0, 1)
        # return z_term - x_term

class ProductLaplaceKernel(Kernel):
    def __init__(self, bandwidth: float, exponent: float, eps: float = 1e-10, bandwidth_mode: str = 'constant'):
        super().__init__()
        assert bandwidth > 0
        assert exponent > 0
        assert eps > 0
        self.bandwidth_mode = bandwidth_mode
        self.base_bandwidth = bandwidth
        self.bandwidth = bandwidth
        self.base_bandwidth = bandwidth
        self.exponent = exponent
        self.eps = eps  # this one is for numerical stability

    def get_sample_batch_size(self, n: int, d: int, scalar_size: int = 4, mem_constant: float = 20) -> int:
        total_memory_possible = torch.cuda.get_device_properties(torch.device('cuda')).total_memory
        curr_mem_use = torch.cuda.memory_allocated()
        available_memory = total_memory_possible - curr_mem_use
        return int(available_memory / (mem_constant*n*scalar_size))

    def _get_kernel_matrix_impl(self, x: torch.Tensor, z: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        kernel_mat = torch.cdist(self._transform_m(x, mat), self._transform_m(z, mat), p=self.exponent)
        kernel_mat.clamp_(min=0)
        if not self.is_adaptive_bandwidth:
            self._adapt_bandwidth(kernel_mat)
        kernel_mat.pow_(self.exponent)
        kernel_mat.mul_(-1./(self.bandwidth**self.exponent))
        kernel_mat.exp_()
        return kernel_mat
    
    def _get_kernel_matrix_categorical_impl(self, x: torch.Tensor, z: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        numerical_indices = self.numerical_indices # (n_num,)
        categorical_indices = self.categorical_indices # List of (d_cat_i,) for each categorical feature
        categorical_vectors = self.categorical_vectors # List of (d_cat_i, d_cat_i) for each categorical feature

        assert len(numerical_indices) > 0 or len(categorical_indices) > 0, "No numerical or categorical features"
        assert len(categorical_indices) == len(categorical_vectors), "Number of categorical index and vector groups must match"
        
        def dist_fn(x, z):
            kernel_mat = torch.cdist(x, z, p=self.exponent)
            kernel_mat.clamp_(min=0)
            if not self.is_adaptive_bandwidth:
                self._adapt_bandwidth(kernel_mat)
            if self.exponent != 1.0:
                kernel_mat.pow_(self.exponent)
            return kernel_mat

        
        mat_num = get_sub_matrix(mat, numerical_indices)
        xnum = self._transform_m(x[:, numerical_indices], mat_num)
        znum = self._transform_m(z[:, numerical_indices], mat_num)

        batch_size = self.get_sample_batch_size(znum.shape[0], znum.shape[1])
        print("Computed batch size", batch_size)
        start_time = time.time()
        dist_mat = torch.zeros((xnum.shape[0], znum.shape[0]), device=xnum.device, dtype=xnum.dtype)
        for i in range(0, xnum.shape[0], batch_size):
            dist_mat[i:i+batch_size, :] = dist_fn(xnum[i:i+batch_size], znum)
        print("Time taken for numerical part", time.time() - start_time)

        # For each categorical feature
        for cat_idx, cat_vecs in zip(categorical_indices, categorical_vectors):
            # Get the kernel matrix for this categorical feature's embeddings
            start_time = time.time()
            mat_cat = get_sub_matrix(mat, cat_idx)
            cat_vecs_transformed = self._transform_m(cat_vecs, mat_cat)
            print("Time taken for categorical embeddings", time.time() - start_time)

            start_time = time.time()
            cat_embedding_kernel = dist_fn(cat_vecs_transformed, cat_vecs_transformed)
            print("Time taken for categorical kernel", time.time() - start_time)

            # Index into the kernel matrix using the categorical indices
            # This creates a matrix of shape (n_x, n_z) with the appropriate kernel values
            # Batch over x_cat to avoid potential memory issues
            start_time = time.time()
            for i in range(0, x.shape[0], batch_size):
                # Index into the kernel matrix using the categorical indices in batches
                # This creates a matrix of shape (batch_size, n_z) with the appropriate kernel values
                # dist_mat[i:i+batch_size].add_(cat_embedding_kernel[x_cat[i:i+batch_size, None], z_cat[None, :]])

                # Use einsum for efficient matrix multiplication
                result = torch.einsum('ij,jk,lk->il', x[i:i+batch_size].float(), cat_embedding_kernel, z.float())
                dist_mat[i:i+batch_size].add_(result)
            print("Time taken for batching", time.time() - start_time)

        print("Time taken for categorical part", time.time() - start_time)

        dist_mat.mul_(-1./(self.bandwidth**self.exponent))
        dist_mat.exp_()
        return dist_mat

    def _get_function_grad_impl(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        def forward_func(z):
            dists = torch.cdist(x, z, p=self.exponent) ** self.exponent
            factor = -((1. / self.bandwidth) ** self.exponent)
            # this is \sum_j f(z_j), so the derivative wrt z will be jacobian(f)(z_j) for all z_j
            return coefs @ torch.exp(factor * (dists * (dists >= self.eps))).sum(dim=1)

        return torch.func.jacrev(forward_func)(z)
    
    def get_agop_categorical(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor,
                 mat: Optional[torch.Tensor] = None, center_grads: bool = False) -> torch.Tensor:
        
        numerical_indices = self.numerical_indices
        categorical_indices = self.categorical_indices

        # see get_function_grads
        f_grads = self.get_function_grads(x, z, coefs, mat)
        # merge output and n_z dims
        f_grads = f_grads.reshape(-1, f_grads.shape[-1])
        if center_grads:
            f_grads = f_grads - f_grads.mean(dim=0, keepdim=True)
        
        # Initialize the final AGOP matrix with zeros
        d = x.shape[1]
        agop = torch.zeros((d, d), device=x.device, dtype=x.dtype)
        
        # Place numerical block if it exists
        if numerical_indices is not None and len(numerical_indices) > 0:
            agop[numerical_indices[:, None], numerical_indices] = (
                f_grads[:, numerical_indices].T @ f_grads[:, numerical_indices]
            )
        
        # Place categorical blocks
        if categorical_indices is not None:
            for cat_idx in categorical_indices:
                agop[cat_idx[:, None], cat_idx] = (
                    f_grads[:, cat_idx].T @ f_grads[:, cat_idx]
                )
        
        return agop


class LpqLaplaceKernel(Kernel):
    def __init__(self, bandwidth: float, p: float, q: float, eps: float = 1e-10, bandwidth_mode: str = 'constant'):
        super().__init__()
        assert bandwidth > 0
        assert 0 < p <= 2
        assert 0 < q <= p
        assert eps > 0
        self.bandwidth = bandwidth
        self.base_bandwidth = bandwidth
        self.p = p
        self.q = q
        self.eps = eps  # this one is for numerical stability
        self.bandwidth_mode = bandwidth_mode

    def _get_kernel_matrix_impl(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.p == 2:
            # faster implementation
            kernel = LaplaceKernel(bandwidth=self.bandwidth, exponent=self.q, eps=self.eps, bandwidth_mode=self.bandwidth_mode)
            return kernel.get_kernel_matrix(x, z)

        kernel_mat = torch.cdist(x, z, p=self.p)
        kernel_mat.clamp_(min=0)
        if not self.is_adaptive_bandwidth:
            self._adapt_bandwidth(kernel_mat)
        kernel_mat.pow_(self.q)
        kernel_mat.mul_(-1. / (self.bandwidth ** self.q))
        kernel_mat.exp_()
        return kernel_mat

    def _get_function_grad_impl(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        if self.p == 2:
            # use faster implementation
            kernel = LaplaceKernel(bandwidth=self.bandwidth, exponent=self.q, eps=self.eps,
                                   bandwidth_mode=self.bandwidth_mode)
            return kernel.get_function_grads(x, z, coefs)

        def forward_func(z):
            dists = torch.cdist(x, z, p=self.p) ** self.q
            factor = -((1. / self.bandwidth) ** self.q)
            # this is \sum_j f(z_j), so the derivative wrt z will be jacobian(f)(z_j) for all z_j
            return coefs @ torch.exp(factor * (dists * (dists >= self.eps))).sum(dim=1)

        return torch.func.jacrev(forward_func)(z)

        # return get_laplacian_gen_grad(z, x, sqrtM=None, v=self.exponent, L=self.bandwidth, alphas=coefs.t(), eps=self.eps).transpose(0, 1)

        # def compute_grad(out_idx: int):
        #     z_cl = z.clone()
        #     z_cl.requires_grad = True
        #     dists = torch.cdist(x, z_cl, p=self.exponent) ** self.exponent
        #     # masking
        #     mask = dists >= self.eps
        #
        #     factor = -((1./self.bandwidth)**self.exponent)
        #
        #     # this is \sum_j f(z_j), so the derivative wrt z will be \nabla f(z_j) for all z_j
        #     sum_f = torch.dot(coefs[out_idx, :], torch.exp(factor * (dists * mask)).sum(dim=1))
        #     sum_f.backward()
        #     return z_cl.grad
        # return torch.stack([compute_grad(i) for i in range(coefs.shape[0])], dim=0)


class SumPowerLaplaceKernel(Kernel):
    def __init__(self, bandwidth: float, exponent: float, eps: float = 1e-10, const_mix: float = 0.0,
                 power: int = 2,
                 bandwidth_mode: str = 'constant'):
        super().__init__()
        assert bandwidth > 0
        assert exponent > 0
        assert eps > 0
        assert 0 <= const_mix < 1
        assert bandwidth_mode == 'constant', 'Adaptive bandwidth currently not supported'
        self.bandwidth = bandwidth
        self.base_bandwidth = bandwidth
        self.exponent = exponent
        self.const_mix = const_mix
        self.power = power
        self.eps = eps  # this one is for numerical stability
        self.bandwidth_mode = bandwidth_mode

    def _get_kernel_matrix_impl(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        diffs = x[:, None, :] - z[None, :, :]
        diffs.abs_()
        diffs.pow_(self.exponent)
        diffs.mul_(-1. / (self.bandwidth ** self.exponent))
        diffs.exp_()
        sum = diffs.sum(dim=-1)
        sum.mul_((1.0 - self.const_mix) / x.shape[1])  # normalize so the max sum is 1
        sum.add_(self.const_mix)
        sum.pow_(self.power)
        return sum

    def _get_function_grad_impl(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        def forward_func(z):
            # compute \sum_j f(z_j)
            diffs = torch.abs(x[:, None, :] - z[None, :, :]).pow(self.exponent)
            diffs = torch.exp((-1. / (self.bandwidth ** self.exponent)) * diffs)
            sum = (1.0 - self.const_mix) * (diffs.sum(dim=-1) / x.shape[-1]) + self.const_mix
            sum = sum ** self.power
            sum = sum.sum(dim=-1)  # sum over z
            return coefs @ sum

        return torch.func.jacrev(forward_func)(z)


if __name__ == '__main__':
    # kernel = LaplaceKernel(bandwidth=2.0, exponent=1.0)
    kernel = ProductLaplaceKernel(bandwidth=2.0, exponent=1.2)

    n_samples = 2000
    n_features = 100
    x = torch.rand(n_samples, n_features)
    coefs = torch.rand(1, n_samples)
    kernel.get_agop(x, x, coefs)

    print('here')

    import matplotlib.pyplot as plt

    x = torch.linspace(-2.0, 2.0, 5)[:, None]
    z = torch.linspace(-4.0, 4.0, 500)[:, None]
    coefs = torch.as_tensor([[1.0, 0.8, 0.4, -0.5, -2.0], [0.1, 0.2, 0.3, 0.4, 0.5]])
    # mat = None
    mat = torch.as_tensor([0.5])
    # mat = torch.as_tensor([[0.5]])
    f_values = coefs[0, :] @ kernel.get_kernel_matrix(x, z, mat)
    plt.plot(z[:, 0], f_values, 'tab:blue', label='function')
    plt.plot(z, kernel.get_function_grads(x, z, coefs, mat)[0], 'tab:orange', label='gradient')
    plt.plot(0.5 * (z[1:, 0] + z[:-1, 0]), (f_values[1:] - f_values[:-1]) / (z[1:, 0] - z[:-1, 0]), color='tab:green',
             linestyle='--', label='finite diff')
    plt.legend()
    plt.show()
