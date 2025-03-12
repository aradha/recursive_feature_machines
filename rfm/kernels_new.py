from typing import Optional

import torch


class Kernel:
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
                x = x * mat[None, :]
            elif len(mat.shape) == 2:
                x = x @ mat
            else:
                raise ValueError(f'm_matrix should have one or two dimensions, but got shape {mat.shape}')
        return x

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
        return self._get_kernel_matrix_impl(self._transform_m(x, mat), self._transform_m(z, mat))

    def get_kernel_matrix_symm(self, x: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        # todo: only compute certain blocks?
        return self.get_kernel_matrix(x, x, mat)

    def get_function_grads(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor,
                           mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return the matrix of function gradients at points z.
        The function is given by \sum_i coefs[i] * k(x[i], \cdot).
        :param x: Matrix of shape (n_x, d_in)
        :param z: Matrix of shape (n_z, d_in)
        :param coefs: Vector of shape (n_x,)
        :param mat: Matrix of shape (d_in, d_out) or vector of shape (d_in)
        :return: Should return a tensor of shape (n_z, d_in).
        """
        grads = self._get_function_grad_impl(self._transform_m(x, mat), self._transform_m(z, mat), coefs)
        return self._transform_m(grads, mat)

    def get_agop(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor,
                 mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        f_grads = self.get_function_grads(x, z, coefs, mat)
        return f_grads.t() @ f_grads

    def get_agop_diag(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor,
                      mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        f_grads = self.get_function_grads(x, z, coefs, mat)
        return f_grads.square().sum(dim=0)


class LaplaceKernel(Kernel):
    def __init__(self, bandwidth: float, exponent: float):
        assert bandwidth > 0
        assert exponent > 0
        self.bandwidth = bandwidth
        self.exponent = exponent

    def _get_kernel_matrix_impl(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        kernel_mat = torch.cdist(x, z)
        kernel_mat.clamp_(min=0)
        if self.exponent != 1.0:
            kernel_mat.pow_(self.exponent)
        kernel_mat.mul_(-1. / self.bandwidth)
        kernel_mat.exp_()
        return kernel_mat

    def _get_function_grad_impl(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(x, z)
        dists.clamp_(min=0)

        # gradient of k(x, z) = exp(-\gamma \|x - z\|^\beta) wrt z  (where \beta = self.exponent)
        # is -\gamma k(x, z) \beta \|x - z\|^{\beta - 1} (z-x)/\|x-z\| = -\gamma \beta k(x, z) \|x - z\|^{\beta-2} (z-x)
        # therefore, setting f(z) = \sum_i coefs[i] k(x[i], z), we have
        # \grad f(z[j]) = \sum_i coefs[i] M[i, j] (z[j] - x[i]),
        # where M[i, j] = -\gamma \beta k(x[i], z[j]) \|x[i] - z[j]\|^{\beta - 2}
        gamma = 1. / self.bandwidth
        kernel_mat = dists ** self.exponent
        kernel_mat.mul_(-gamma)
        kernel_mat.exp_()

        # now compute M
        # todo: is this good enough for the masking?
        dists.clamp_(min=1e-8)  # todo: make configurable?
        dists.pow_(self.exponent - 2)
        kernel_mat.mul_(dists)
        kernel_mat.mul_(-gamma * self.exponent)

        # now we want result[j, d] = \sum_i coefs[i] grad_mat[i, j] (z[j, d] - x[i, d])
        kernel_mat.mul_(coefs[:, None])

        return kernel_mat.sum(dim=0)[:, None] * z - kernel_mat.t() @ x


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = torch.linspace(-2.0, 2.0, 5)[:, None]
    z = torch.linspace(-4.0, 4.0, 500)[:, None]
    coefs = torch.as_tensor([1.0, 0.8, 0.4, -0.5, -2.0])
    kernel = LaplaceKernel(bandwidth=2.0, exponent=1.2)
    # mat = None
    mat = torch.as_tensor([0.5])
    # mat = torch.as_tensor([[0.5]])
    f_values = coefs @ kernel.get_kernel_matrix(x, z, mat)
    plt.plot(z[:, 0], f_values, 'tab:blue', label='function')
    plt.plot(z, kernel.get_function_grads(x, z, coefs, mat), 'tab:orange', label='gradient')
    plt.plot(0.5 * (z[1:, 0] + z[:-1, 0]), (f_values[1:] - f_values[:-1]) / (z[1:, 0] - z[:-1, 0]), color='tab:green',
             linestyle='--', label='finite diff')
    plt.legend()
    plt.show()
