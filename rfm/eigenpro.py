'''Construct kernel model with EigenPro optimizer.'''
import collections
import time
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from .svd import nystrom_kernel_svd

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

    
    eigvals, eigvecs = nystrom_kernel_svd(samples, map_fn, svd_q)

    # Choose k such that the batch size is bounded by
    #   the subsample size and the memory size.
    #   Keep the original k if it is pre-specified.
    if top_q is None:
        max_bs = min(max(n_sample / 5, bs_gpu), n_sample)
        top_q = torch.sum(torch.pow(1 / eigvals, alpha) < max_bs) - 1
        top_q = max(top_q, min_q)

    print("top_q", top_q, "svd_q", svd_q)
    eigvals, tail_eigval = eigvals[:top_q - 1], eigvals[top_q - 1]
    eigvecs = eigvecs[:, :top_q - 1]

    device = samples.device
    eigvals_t = eigvals.to(device)
    eigvecs_t = eigvecs.to(device)
    tail_eigval_t = torch.tensor(tail_eigval, dtype=samples.dtype).to(device)

    scale = torch.pow(eigvals[0] / tail_eigval, alpha).to(samples.dtype)
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
    knorms = 1 - torch.sum(eigvecs ** 2, dim=1) * n_sample
    beta = torch.max(knorms)

    return eigenpro_fn, scale.item(), eigvals[0].item(), beta.to(samples.dtype).item()


class KernelModel(nn.Module):
    '''Fast Kernel Regression using EigenPro iteration.'''
    def __init__(self, kernel_fn, centers, y_dim, device="cuda"):
        super(KernelModel, self).__init__()
        self.kernel_fn = kernel_fn
        self.n_centers, self.x_dim = centers.shape
        self.device = device
        self.pinned_list = []

        self.centers = self.tensor(centers, release=True, dtype=centers.dtype)
        self.weight = self.tensor(torch.zeros(
            self.n_centers, y_dim), release=True, dtype=centers.dtype)

    def __del__(self):
        for pinned in self.pinned_list:
            _ = pinned.to("cpu")

    def tensor(self, data, dtype=None, release=False):
        tensor = torch.tensor(data, requires_grad=False, device=self.device)

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
        return bs, float(eta)

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

    def evaluate(self, X_eval, y_eval, bs,
                 metrics=('mse')):
        
        p_list = []
        n_sample, _ = X_eval.shape
        n_batch = n_sample / min(n_sample, bs)
        for batch_ids in np.array_split(range(n_sample), n_batch):
            x_batch = self.tensor(X_eval[batch_ids], dtype=X_eval.dtype)
            p_batch = self.forward(x_batch).cpu()
            p_list.append(p_batch)
        p_eval = torch.concat(p_list, dim=0)

        eval_metrics = collections.OrderedDict()
        if 'mse' in metrics:
            eval_metrics['mse'] = torch.mean(torch.square(p_eval - y_eval.cpu())).item()
        if 'multiclass-acc' in metrics:
            y_class = torch.argmax(y_eval, dim=-1)
            p_class = torch.argmax(p_eval, dim=-1)
            eval_metrics['multiclass-acc'] = torch.sum(y_class == p_class).item() / n_sample
        if 'binary-acc' in metrics:
            y_class = torch.where(y_eval > 0.5, 1, 0).reshape(-1)
            p_class = torch.where(p_eval > 0.5, 1, 0).reshape(-1)
            eval_metrics['binary-acc'] = torch.sum(y_class == p_class).item() / n_sample
        if 'f1' in metrics:
            y_class = torch.where(y_eval > 0.5, 1, 0).reshape(-1)
            p_class = torch.where(p_eval > 0.5, 1, 0).reshape(-1)
            eval_metrics['f1'] = torch.mean(2 * (y_class * p_class) / (y_class + p_class + 1e-8)).item()
        if 'auc' in metrics:
            eval_metrics['auc'] = roc_auc_score(y_eval.cpu().flatten(), p_eval.cpu().flatten())

        return eval_metrics

    def fit(self, X_train, y_train, X_val, y_val, epochs, mem_gb,
            n_subsamples=None, top_q=None, bs=None, eta=None,
            n_train_eval=5000, run_epoch_eval=True, lr_scale=1, seed=1, classification=False):
        
        n_train_eval = min(bs, n_train_eval)
        metrics = ('mse',)
        if classification:
            if y_train.shape[-1] == 1:
                metrics += ('binary-acc', 'f1', 'auc')
            else:
                metrics += ('multiclass-acc')

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
        eta = self.tensor(lr_scale * eta / bs, dtype=X_train.dtype)

        # Subsample training data for fast estimation of training loss.
        ids = np.random.choice(n_samples,
                               min(n_samples, n_train_eval),
                               replace=False)
        X_train_eval, y_train_eval = X_train[ids], y_train[ids]

        res = dict()
        initial_epoch = 0
        train_sec = 0  # training time in seconds
        best_weights = None
        if classification:
            best_metric = 0
        else:
            best_metric = float('inf')

        for epoch in range(epochs):
            start = time.time()
            for _ in range(epoch - initial_epoch):
                epoch_ids = np.random.choice(
                    n_samples, n_samples // bs * bs, replace=False)
                for batch_ids in tqdm(np.array_split(epoch_ids, n_samples / bs)):
                    x_batch = self.tensor(X_train[batch_ids], dtype=X_train.dtype)
                    y_batch = self.tensor(y_train[batch_ids], dtype=y_train.dtype)
                    batch_ids = self.tensor(batch_ids)
                    self.eigenpro_iterate(samples, x_batch, y_batch, eigenpro_f,
                                          eta, sample_ids, batch_ids)
                    del x_batch, y_batch, batch_ids

            if run_epoch_eval:
                train_sec += time.time() - start
                tr_score = self.evaluate(X_train_eval, y_train_eval, bs, metrics=metrics)
                tv_score = self.evaluate(X_val, y_val, bs, metrics=metrics)
                out_str = f"({epoch} epochs, {train_sec} seconds)\t train l2: {tr_score['mse']} \tval l2: {tv_score['mse']}"
                if classification:
                    if 'binary-acc' in tr_score:
                        out_str += f"\t train acc: {tr_score['binary-acc']} \tval acc: {tv_score['binary-acc']}"
                    else:
                        out_str += f"\t train acc: {tr_score['multiclass-acc']} \tval acc: {tv_score['multiclass-acc']}"
                    if 'f1' in tr_score:
                        out_str += f"\t train f1: {tr_score['f1']} \tval f1: {tv_score['f1']}"
                    if 'auc' in tr_score:
                        out_str += f"\t train auc: {tr_score['auc']} \tval auc: {tv_score['auc']}"
                print(out_str)
                res[epoch] = (tr_score, tv_score, train_sec)
                if classification:
                    if 'auc' in tv_score:
                        if tv_score['auc'] > best_metric:
                            best_metric = tv_score['auc']
                            best_weights = self.weight.cpu().clone()
                            print(f"New best auc: {best_metric}")
                    elif 'binary-acc' in tv_score:
                        if tv_score['binary-acc'] > best_metric:
                            best_metric = tv_score['binary-acc']
                            best_weights = self.weight.cpu().clone()
                            print(f"New best binary-acc: {best_metric}")
                    elif 'multiclass-acc' in tv_score:
                        if tv_score['multiclass-acc'] > best_metric:
                            best_metric = tv_score['multiclass-acc']
                            best_weights = self.weight.cpu().clone()
                            print(f"New best multiclass-acc: {best_metric}")
                    else:
                        if tv_score['mse'] < best_metric:
                            best_metric = tv_score['mse']
                            best_weights = self.weight.cpu().clone()
                            print(f"New best mse: {best_metric}")
            initial_epoch = epoch

        self.weight = best_weights.to(self.device)

        return res
