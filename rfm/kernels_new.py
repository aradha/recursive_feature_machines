from typing import Optional

import torch


class Kernel:
    def _get_kernel_matrix_impl(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def _get_kernel_grad_tensor_impl(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
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

    def get_kernel_grad_tensor(self, x: torch.Tensor, z: torch.Tensor,
                               mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return the tensor of kernel matrix gradients wrt the second argument.
        :param x: Matrix of shape (n_x, d_in)
        :param z: Matrix of shape (n_z, d_in)
        :param mat: Matrix of shape (d_in, d_out) or vector of shape (d_in)
        :return: Should return a tensor of shape (n_x, n_z, d_in).
        """
        raw_deriv_tensor = self._get_kernel_grad_tensor_impl(self._transform_m(x, mat),
                                                             self._transform_m(z, mat))

        if mat is not None:
            if len(mat.shape) == 1:
                return raw_deriv_tensor * mat[None, None, :]
            elif len(mat.shape) == 2:
                return torch.einsum('xzd,di->xzi', raw_deriv_tensor, mat)
            else:
                raise ValueError(f'm_matrix should have one or two dimensions, but got shape {mat.shape}')

        return raw_deriv_tensor

    def get_function_grads(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor,
                           mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return the matrix of function gradients at points z. The function is given by \sum_i coefs[i] * k(x[i], \cdot).
        :param x:
        :param z:
        :param coefs:
        :param mat: sqrtM matrix.
        :return:
        """
        # optimization: don't apply m_matrix inside the grad tensor computation,
        # only apply it after summing over coefficients
        deriv_tensor = self.get_kernel_grad_tensor(self._transform_m(x, mat), self._transform_m(z, mat))
        # gradients of the function at points z
        return self._transform_m(torch.einsum('x,xzd->zd', coefs, deriv_tensor), mat)

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
        kernel_mat.mul_(-1./self.bandwidth)
        kernel_mat.exp_()
        return kernel_mat

    def _get_func_grad_impl(self, x: torch.Tensor, z: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(x, z)
        dists.clamp_(min=0)

        # gradient of k(x, z) = exp(-\gamma \|x - z\|^\beta) wrt z  (where \beta = self.exponent)
        # is -\gamma k(x, z) \beta \|x - z\|^{\beta - 1} (z-x)/\|x-z\| = -\gamma \beta k(x, z) \|x - z\|^{\beta-2} (z-x)
        # therefore, setting f(z) = \sum_i coefs[i] k(x[i], z), we have
        # \grad f(z[j]) = \sum_i coefs[i] M[i, j] (z[j] - x[i]),
        # where M[i, j] = -\gamma \beta k(x[i], z[j]) \|x[i] - z[j]\|^{\beta - 2}
        gamma = 1./self.bandwidth
        kernel_mat = dists ** self.exponent
        kernel_mat.mul_(-gamma)
        kernel_mat.exp_()

        # now compute M
        dists.clamp_(min=1e-8)  # todo: make configurable?
        dists.pow_(self.exponent-2)
        kernel_mat.mul_(dists)
        kernel_mat.mul_(-gamma*self.exponent)

        # now we want result[j, d] = \sum_i coefs[i] grad_mat[i, j] (z[j, d] - x[i, d])
        kernel_mat.mul_(coefs[:, None])

        return kernel_mat.sum(dim=0)[:, None] * z - kernel_mat @ x


