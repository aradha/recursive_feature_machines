'''Construct kernel model with EigenPro optimizer.'''
import collections
import time
import torch

import torch.nn as nn
import numpy as np

import svd
import utils


def asm_eigenpro_fn(samples, map_fn, top_q, bs_gpu, alpha, min_q=5, seed=1):
    """Prepare gradient map for EigenPro and calculate
    scale factor for learning ratesuch that the update rule,
        p <- p - eta * g
    becomes,
        p <- p - scale * eta * (g - eigenpro_fn(g))

    Arguments:
        samples:	matrix of shape (n_sample, n_feature).
        map_fn:    	kernel k(samples, centers) where centers are specified.
        top_q:  	top-q eigensystem for constructing eigenpro iteration/kernel.
        bs_gpu:     maxinum batch size corresponding to GPU memory.
        alpha:  	exponential factor (<= 1) for eigenvalue rescaling due to approximation.
        min_q:  	minimum value of q when q (if None) is calculated automatically.
        seed:   	seed for random number generation.

    Returns:
        eigenpro_fn:	tensor function.
        scale:  		factor that rescales learning rate.
        top_eigval:  	largest eigenvalue.
        beta:   		largest k(x, x) for the EigenPro kernel.
    """

    np.random.seed(seed)  # set random seed for subsamples
    start = time.time()
    n_sample, _ = samples.shape

    if top_q is None:
        svd_q = min(n_sample - 1, 1000)
    else:
        svd_q = top_q

    eigvals, eigvecs = svd.nystrom_kernel_svd(samples, map_fn, svd_q)

    # Choose k such that the batch size is bounded by
    #   the subsample size and the memory size.
    #   Keep the original k if it is pre-specified.
    if top_q is None:
        max_bs = min(max(n_sample / 5, bs_gpu), n_sample)
        top_q = np.sum(np.power(1 / eigvals, alpha) < max_bs) - 1
        top_q = max(top_q, min_q)

    eigvals, tail_eigval = eigvals[:top_q - 1], eigvals[top_q - 1]
    eigvecs = eigvecs[:, :top_q - 1]

    device = samples.device
    eigvals_t = torch.tensor(eigvals.copy()).to(device)
    eigvecs_t = torch.tensor(eigvecs).to(device)
    tail_eigval_t = torch.tensor(tail_eigval, dtype=torch.float).to(device)

    scale = utils.float_x(np.power(eigvals[0] / tail_eigval, alpha))
    diag_t = (1 - torch.pow(tail_eigval_t / eigvals_t, alpha)) / eigvals_t

    def eigenpro_fn(grad, kmat):
        '''Function to apply EigenPro preconditioner.'''
        return torch.mm(eigvecs_t * diag_t,
                        torch.t(torch.mm(torch.mm(torch.t(grad),
                                                  kmat),
                                         eigvecs_t)))

    print("SVD time: %.2f, top_q: %d, top_eigval: %.2f, new top_eigval: %.2e" %
          (time.time() - start, top_q, eigvals[0], eigvals[0] / scale))

    #beta = kmat.diag().max()
    knorms = 1 - np.sum(eigvecs ** 2, axis=1) * n_sample
    beta = np.max(knorms)

    return eigenpro_fn, scale, eigvals[0], utils.float_x(beta)


class FKR_EigenPro(nn.Module):
    '''Fast Kernel Regression using EigenPro iteration.'''
    def __init__(self, kernel_fn, centers, y_dim, device="cuda"):
        super(FKR_EigenPro, self).__init__()
        self.kernel_fn = kernel_fn
        self.n_centers, self.x_dim = centers.shape
        self.device = device
        self.pinned_list = []

        self.centers = self.tensor(centers, release=True)
        self.weight = self.tensor(torch.zeros(
            self.n_centers, y_dim), release=True)

    def __del__(self):
        for pinned in self.pinned_list:
            _ = pinned.to("cpu")
        torch.cuda.empty_cache()

    def tensor(self, data, dtype=None, release=False):
        tensor = torch.tensor(data, dtype=dtype,
                              requires_grad=False).to(self.device)
        if release:
            self.pinned_list.append(tensor)
        return tensor

    def kernel_matrix(self, samples):
        return self.kernel_fn(samples, self.centers)

    def forward(self, samples, weight=None):
        if weight is None:
            weight = self.weight
        kmat = self.kernel_matrix(samples)
        pred = kmat.mm(weight)
        return pred

    def primal_gradient(self, samples, labels, weight):
        pred = self.forward(samples, weight)
        grad = pred - labels
        return grad

    @staticmethod
    def _compute_opt_params(bs, bs_gpu, beta, top_eigval):
        if bs is None:
            bs = min(np.int32(beta / top_eigval + 1), bs_gpu)

        if bs < beta / top_eigval + 1:
            eta = bs / beta
        else:
            eta = 0.99 * 2 * bs / (beta + (bs - 1) * top_eigval)
        return bs, utils.float_x(eta)

    def eigenpro_iterate(self, samples, x_batch, y_batch, eigenpro_fn,
                         eta, sample_ids, batch_ids):
        # update random coordiate block (for mini-batch)
        grad = self.primal_gradient(x_batch, y_batch, self.weight)
        self.weight.index_add_(0, batch_ids, -eta * grad)

        # update fixed coordinate block (for EigenPro)
        kmat = self.kernel_fn(x_batch, samples)
        correction = eigenpro_fn(grad, kmat)
        self.weight.index_add_(0, sample_ids, eta * correction)
        return

    def evaluate(self, x_eval, y_eval, bs,
                 metrics=('mse', 'multiclass-acc')):
        p_list = []
        n_sample, _ = x_eval.shape
        n_batch = n_sample / min(n_sample, bs)
        for batch_ids in np.array_split(range(n_sample), n_batch):
            x_batch = self.tensor(x_eval[batch_ids])
            p_batch = self.forward(x_batch).cpu().data.numpy()
            p_list.append(p_batch)
        p_eval = np.vstack(p_list)

        eval_metrics = collections.OrderedDict()
        if 'mse' in metrics:
            eval_metrics['mse'] = np.mean(np.square(p_eval - y_eval))
        if 'multiclass-acc' in metrics:
            y_class = np.argmax(y_eval, axis=-1)
            p_class = np.argmax(p_eval, axis=-1)
            eval_metrics['multiclass-acc'] = np.mean(y_class == p_class)

        return eval_metrics

    def fit(self, x_train, y_train, x_val, y_val, epochs, mem_gb,
            n_subsamples=None, top_q=None, bs=None, eta=None,
            n_train_eval=5000, run_epoch_eval=True, scale=1, seed=1):

        n_samples, n_labels = y_train.shape
        if n_subsamples is None:
            if n_samples < 100000:
                n_subsamples = min(n_samples, 2000)
            else:
                n_subsamples = 12000

        mem_bytes = (mem_gb - 1) * 1024**3  # preserve 1GB
        bsizes = np.arange(n_subsamples)
        mem_usages = ((self.x_dim + 3 * n_labels + bsizes + 1)
                      * self.n_centers + n_subsamples * 1000) * 4
        bs_gpu = np.sum(mem_usages < mem_bytes)  # device-dependent batch size

        # Calculate batch size / learning rate for improved EigenPro iteration.
        np.random.seed(seed)
        sample_ids = np.random.choice(n_samples, n_subsamples, replace=False)
        sample_ids = self.tensor(sample_ids)
        samples = self.centers[sample_ids]
        eigenpro_f, gap, top_eigval, beta = asm_eigenpro_fn(
            samples, self.kernel_fn, top_q, bs_gpu, alpha=.95, seed=seed)
        new_top_eigval = top_eigval / gap

        if eta is None:
            bs, eta = self._compute_opt_params(
                bs, bs_gpu, beta, new_top_eigval)
        else:
            bs, _ = self._compute_opt_params(bs, bs_gpu, beta, new_top_eigval)

        print("n_subsamples=%d, bs_gpu=%d, eta=%.2f, bs=%d, top_eigval=%.2e, beta=%.2f" %
              (n_subsamples, bs_gpu, eta, bs, top_eigval, beta))
        eta = self.tensor(scale * eta / bs, dtype=torch.float)

        # Subsample training data for fast estimation of training loss.
        ids = np.random.choice(n_samples,
                               min(n_samples, n_train_eval),
                               replace=False)
        x_train_eval, y_train_eval = x_train[ids], y_train[ids]

        res = dict()
        initial_epoch = 0
        train_sec = 0  # training time in seconds

        for epoch in epochs:
            start = time.time()
            for _ in range(epoch - initial_epoch):
                epoch_ids = np.random.choice(
                    n_samples, n_samples // bs * bs, replace=False)
                for batch_ids in np.array_split(epoch_ids, n_samples / bs):
                    x_batch = self.tensor(x_train[batch_ids])
                    y_batch = self.tensor(y_train[batch_ids])
                    batch_ids = self.tensor(batch_ids)
                    self.eigenpro_iterate(samples, x_batch, y_batch, eigenpro_f,
                                          eta, sample_ids, batch_ids)
                    del x_batch, y_batch, batch_ids

            if run_epoch_eval:
                train_sec += time.time() - start
                tr_score = self.evaluate(x_train_eval, y_train_eval, bs)
                tv_score = self.evaluate(x_val, y_val, bs)
                print("train error: %.2f%%\tval error: %.2f%% (%d epochs, %.2f seconds)\t"
                      "train l2: %.2e\tval l2: %.2e" %
                      ((1 - tr_score['multiclass-acc']) * 100,
                      (1 - tv_score['multiclass-acc']) * 100,
                      epoch, train_sec, tr_score['mse'], tv_score['mse']))
                res[epoch] = (tr_score, tv_score, train_sec)

            initial_epoch = epoch

        return res
