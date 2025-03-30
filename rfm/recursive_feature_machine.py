from .eigenpro import KernelModel
    
import torch, numpy as np
from torchmetrics.functional.classification import accuracy
from .generic_kernels import Kernel, LaplaceKernel, ProductLaplaceKernel
from .kernels import laplacian_M, gaussian_M, euclidean_distances_M, laplacian_gen, get_laplace_gen_agop, ntk_kernel
from tqdm.contrib import tenumerate
from .utils import matrix_power, get_data_from_loader
import time

class RecursiveFeatureMachine(torch.nn.Module):
    """
    Main object for RFMs with sklearn style interface. Subclasses must implement the kernel and update_M methods. 
    The subclasses may be either specific kernels (Laplace, Gaussian, GeneralizedLaplace, etc.), in which case the kernel method is automatically derived,
    or generic kernels (GenericKernel), in which case a Kernel object must be provided. I.e. one can either define:
    ```python
        from rfm import LaplaceRFM
        model = LaplaceRFM(bandwidth=1, device='cpu', reg=1e-3, iters=3, bandwidth_mode='constant')

        or

        from rfm import GenericRFM
        from rfm.kernels_new import LaplaceKernel
        model = GenericRFM(kernel=LaplaceKernel(bandwidth=1, exponent=1.2), device='cpu', reg=1e-3, iters=3, bandwidth_mode='constant')
    ```
    """

    def __init__(self, device=None, mem_gb=8, diag=False, centering=False, reg=1e-3, iters=5, p_batch_size=None, bandwidth_mode='constant'):
        """
        :param device: device to run the model on
        :param mem_gb: memory in GB for AGOP
        :param diag: if True, Mahalanobis matrix M will be diagonal
        :param centering: if True, update_M will center the gradients before taking an outer product
        :param reg: regularization for the kernel matrix
        :param iters: number of iterations to run
        :param p_batch_size: batch size over centers for AGOP computation
        :param bandwidth_mode: 'constant' or 'adaptive'
        """
        super().__init__()
        self.M = None
        self.sqrtM = None
        self.use_sqrtM = False
        self.model = None
        self.diag = diag # if True, Mahalanobis matrix M will be diagonal
        self.centering = centering # if True, update_M will center the gradients before taking an outer product
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.mem_gb = mem_gb
        self.reg = reg # only used when fit with direct solve
        self.iters = iters
        self.kernel_type = None
        self.p_batch_size = p_batch_size
        self.agop_power = 0.5 # power for root of agop
        self.bandwidth_mode = bandwidth_mode
        self.max_lstsq_size = 70_000 # max number of points to use for direct solve

    def kernel(self, x, z):
        raise NotImplementedError("Must implement this method in a subclass")
    
    def update_M(self):
        raise NotImplementedError("Must implement this method in a subclass")
    
    def reset_adaptive_bandwidth(self):
        if self.kernel_type != 'generic':
            raise ValueError("Cannot reset bandwidth for non-generic kernels, choose constant bandwidth mode")
        self.kernel_obj._reset_adaptive_bandwidth()
        return 

    def tensor_copy(self, tensor):
        """
        Create a CPU copy of a tensor.
        :param tensor: Tensor to copy.
        :param keep_device: If True, the device of the original tensor is kept.
        :return: CPU copy of the tensor.
        """
        if tensor is None:
            return None
        elif self.keep_device or tensor.device.type == 'cpu':
            return tensor.clone()
        else:
            return tensor.cpu()
        
    def set_categorical_indices(self, numerical_indices, categorical_indices, categorical_vectors, device=None):
        """
        :param numerical_indices: torch.Tensor(n_num,)
        :param categorical_indices: List of torch.Tensor(d_cat_i,) for each categorical feature
        :param categorical_vectors: List of torch.Tensor(d_cat_i, d_cat_i) for each categorical feature. Each row is the encoding for that index.
        """
        if self.kernel_type != 'generic' or not isinstance(self.kernel_obj, ProductLaplaceKernel):
            raise ValueError("Can only set categorical indices for generic kernels with ProductLaplaceKernel")
        assert len(categorical_indices) == len(categorical_vectors), "Number of categorical index and vector groups must match"
        assert len(numerical_indices) > 0 or len(categorical_indices) > 0, "No numerical or categorical features"

        self.kernel_obj.set_categorical_indices(numerical_indices, categorical_indices, categorical_vectors, device=self.device if device is None else device)
        return

    def update_best_params(self, best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth, current_metric, current_iter):
        # if classification and accuracy higher, or if regression and mse lower
        if self.classification and current_metric > best_metric:
            best_metric = current_metric
            best_alphas = self.tensor_copy(self.weights)
            best_iter = current_iter
            best_bandwidth = self.bandwidth if self.kernel_type != 'generic' else self.kernel_obj.bandwidth+0
            best_M = self.tensor_copy(self.M)
            best_sqrtM = self.tensor_copy(self.sqrtM)

        elif not self.classification and current_metric < best_metric:
            best_metric = current_metric
            best_alphas = self.tensor_copy(self.weights)
            best_iter = current_iter
            best_bandwidth = self.bandwidth if self.kernel_type != 'generic' else self.kernel_obj.bandwidth+0
            best_M = self.tensor_copy(self.M)
            best_sqrtM = self.tensor_copy(self.sqrtM)

        return best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth
        
    def fit_predictor(self, centers, targets, classification=False, 
                      bs=None, lr_scale=1, verbose=True, solver='solve', **kwargs):
        
        if self.bandwidth_mode == 'adaptive':
            # adaptive bandwidth will be reset on next kernel computation
            print("Resetting adaptive bandwidth")
            self.reset_adaptive_bandwidth()

        self.centers = centers

        if self.fit_using_eigenpro:
            if self.prefit_eigenpro:
                random_indices = torch.randperm(centers.shape[0])[:self.max_lstsq_size]
                start = time.time()
                print(f"Prefitting Eigenpro with {len(random_indices)} points")
                sub_weights = self.fit_predictor_lstsq(centers[random_indices], targets[random_indices], solver=solver)
                end = time.time()
                print(f"Time taken to prefit Eigenpro with {len(random_indices)} points: {end-start} seconds")
                initial_weights = torch.zeros_like(targets)
                initial_weights[random_indices] = sub_weights.to(targets.device, dtype=targets.dtype)
            else:
                initial_weights = None

            self.weights = self.fit_predictor_eigenpro(centers, targets, bs=bs, lr_scale=lr_scale, 
                                                       verbose=verbose, classification=classification, 
                                                       initial_weights=initial_weights,
                                                       **kwargs)
        else:
            self.weights = self.fit_predictor_lstsq(centers, targets, solver=solver)

    def fit_predictor_lstsq(self, centers, targets, solver='solve'):
        assert(len(centers)==len(targets))

        if centers.device != self.device:
            centers = centers.to(self.device)
            targets = targets.to(self.device)

        kernel_matrix = self.kernel(centers, centers)    

        if self.reg > 0:
            kernel_matrix.diagonal().add_(self.reg)
        
        
        if solver == 'solve':
            out = torch.linalg.solve(kernel_matrix, targets)
        elif solver == 'cholesky':
            L = torch.linalg.cholesky(kernel_matrix, out=kernel_matrix)
            out = torch.cholesky_solve(targets, L)
        elif solver == 'lu':
            P, L, U = torch.linalg.lu(kernel_matrix)
            out = torch.linalg.lu_solve(P, L, U, targets)
        else:
            raise ValueError(f"Invalid solver: {solver}")
        
        return out

    def fit_predictor_eigenpro(self, centers, targets, bs, lr_scale, verbose, initial_weights=None, **kwargs):
        n_classes = 1 if targets.dim()==1 else targets.shape[-1]
        ep_model = KernelModel(self.kernel, centers, n_classes, device=self.device)
        if initial_weights is not None:
            ep_model.weight = initial_weights.to(ep_model.weight.device, dtype=ep_model.weight.dtype)
        _ = ep_model.fit(centers, targets, verbose=verbose, mem_gb=self.mem_gb, bs=bs, lr_scale=lr_scale, **kwargs)
        return ep_model.weight.clone()


    def predict(self, samples):
        out = self.kernel(samples.to(self.device), self.centers.to(self.device)) @ self.weights.to(self.device)
        return out.to(samples.device)


    def fit(self, train_data, test_data, iters=None, method='lstsq', 
            classification=True, verbose=True, M_batch_size=None, 
            return_best_params=False, bs=None, 
            return_Ms=False, lr_scale=1, total_points_to_sample=20000, 
            solver='solve', fit_last_M=False, prefit_eigenpro=True, 
            **kwargs):
        """
        :param train_data: torch.utils.data.DataLoader or tuple of (X, y)
        :param test_data: torch.utils.data.DataLoader or tuple of (X, y)
        :param iters: number of iterations to run
        :param method: 'lstsq' or 'eigenpro'
        :param classification: if True, the model will tune for (and report) accuracy, else just MSE loss
        :param verbose: if True, print progress
        :param M_batch_size: batch size over samples for AGOP computation
        :param return_best_params: if True, return the best parameters
        :param bs: batch size for eigenpro
        :param return_Ms: if True, return the Mahalanobis matrix at each iteration
        :param lr_scale: learning rate scale for EigenPro
        :param total_points_to_sample: number of points to sample for AGOP computation
        :param solver: 'solve' or 'cholesky' or 'lu', used in LSTSQ computation
        :param fit_last_M: if True, fit the Mahalanobis matrix one last time after training
        :param prefit_eigenpro: if True, prefit EigenPro with a subset of <= max_lstsq_size samples
        """

        self.verbose = verbose
        self.fit_using_eigenpro = (method.lower()=='eigenpro')
        self.prefit_eigenpro = prefit_eigenpro
        self.use_sqrtM = self.kernel_type in ['laplacian_gen', 'generic']
        self.classification = classification

        if iters is None:
            iters = self.iters

        if isinstance(train_data, torch.utils.data.DataLoader):
            print("Loaders provided")
            X_train, y_train = get_data_from_loader(train_data)
            X_test, y_test = get_data_from_loader(test_data)
        else:
            X_train, y_train = train_data
            X_test, y_test = test_data

        X_train = X_train.to(self.device)
        X_test = X_test.to(self.device)
        y_train = y_train.to(self.device)
        y_test = y_test.to(self.device)

        self.keep_device = X_train.shape[1] > X_train.shape[0] # keep previous Ms on GPU if more features than samples

        metrics, Ms = [], []
        best_alphas, best_M, best_sqrtM = None, None, None
        best_metric = float('inf') if not classification else 0 
        best_iter = None
        best_bandwidth = self.bandwidth if self.kernel_type != 'generic' else self.kernel_obj.bandwidth+0
        for i in range(iters):
            self.fit_predictor(X_train, y_train, X_val=X_test, y_val=y_test, 
                               classification=classification,
                               bs=bs, lr_scale=lr_scale, 
                               verbose=verbose, solver=solver, 
                               **kwargs)
                        
            if classification:
                test_metrics = self.score(X_test, y_test, metrics=['mse', 'accuracy'])
                test_acc = test_metrics['accuracy']
                test_mse = test_metrics['mse']
                if method == 'lstsq':
                    train_metrics = self.score(X_train, y_train, metrics=['mse', 'accuracy'])
                    train_acc = train_metrics['accuracy']
                    if verbose:
                        print(f"Round {i}, Train Acc: {100*train_acc:.2f}%, Test Acc: {100*test_acc:.2f}%")
                else:
                    if verbose:
                        print(f"Round {i}, Test Acc: {100*test_acc:.2f}%")
            else:
                test_mse = self.score(X_test, y_test, metrics=['mse'])['mse']

            if verbose:
                print(f"Round {i}, Test MSE: {test_mse:.4f}")

            # if classification and accuracy higher, or if regression and mse lower
            if return_best_params:
                best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth = self.update_best_params(best_metric, best_alphas, 
                                                                                                                best_M, best_sqrtM, 
                                                                                                                best_iter, best_bandwidth, 
                                                                                                                test_acc if classification else test_mse, i)
                

            self.fit_M(X_train, y_train, verbose=verbose, M_batch_size=M_batch_size, 
                       use_sqrtM=self.use_sqrtM, total_points_to_sample=total_points_to_sample, 
                       **kwargs)
            
            del self.weights
            
            if return_Ms:
                Ms.append(self.tensor_copy(self.M))
                metrics.append(test_acc if classification else test_mse)

        self.fit_predictor(X_train, y_train, X_val=X_test, y_val=y_test, 
                           verbose=verbose, classification=classification, 
                           bs=bs, **kwargs)        
        if classification:
            final_test_metrics = self.score(X_test, y_test, metrics=['mse', 'accuracy'])
            final_test_acc = final_test_metrics['accuracy']
            final_mse = final_test_metrics['mse']
            if verbose:
                print(f"Final Test Acc: {100*final_test_acc:.2f}%")
        else:
            final_mse = self.score(X_test, y_test, metrics=['mse'])['mse']

        if verbose:
            print(f"Final MSE: {final_mse:.4f}")

        if return_best_params:
            best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth = self.update_best_params(best_metric, best_alphas, best_M, 
                                                                                                                best_sqrtM, best_iter, best_bandwidth, 
                                                                                                                final_test_acc if classification else final_mse, 
                                                                                                                iters)
            print(f"Returning best parameters with value: {best_metric:.4f}")
            self.M = None if best_M is None else best_M.to(self.device)
            if self.use_sqrtM and best_sqrtM is not None:
                self.sqrtM = best_sqrtM.to(self.device)
            else:
                self.sqrtM = None
            self.weights = best_alphas.to(self.device)
            if self.kernel_type == 'generic':
                self.kernel_obj.bandwidth = best_bandwidth
            else:
                self.bandwidth = best_bandwidth

        self.best_iter = best_iter
        if fit_last_M:
            self.fit_M(X_train, y_train, verbose=verbose, M_batch_size=M_batch_size, use_sqrtM=self.use_sqrtM, 
                        total_points_to_sample=total_points_to_sample, fit_last_M=fit_last_M, **kwargs)
            Ms.append(self.tensor_copy(self.M))

        if return_Ms and fit_last_M:
            self.agop_best_model = Ms[best_iter]

        if return_Ms:
            return Ms, metrics
            
        return final_mse
    
    def _compute_optimal_M_batch(self, n, c, d, scalar_size=4, mem_constant=2):
        if self.device in ['cpu', torch.device('cpu')]:
            return n 
        
        """Computes the optimal batch size for AGOP."""
        total_memory_possible = torch.cuda.get_device_properties(self.device).total_memory
        curr_mem_use = torch.cuda.memory_allocated()
        available_memory = total_memory_possible - curr_mem_use
        M_batch_size = int(available_memory / (mem_constant*n*c*d*scalar_size))
        print("Optimal M batch size: ", M_batch_size)
        return M_batch_size
    
    def fit_M(self, samples, labels, p_batch_size=None, M_batch_size=None, 
              verbose=True, total_points_to_sample=20000, use_sqrtM=False, **kwargs):
        """Applies AGOP to update the Mahalanobis matrix M."""
        
        n, d = samples.shape
        M = torch.zeros_like(self.M) if self.M is not None else (
            torch.zeros(d, dtype=samples.dtype, device=self.device) 
            if self.diag else torch.zeros(d, d, dtype=samples.dtype, device=self.device))
        

        if M_batch_size is None: 
            BYTES_PER_SCALAR = samples.element_size()
            c = labels.shape[-1]
            M_batch_size = self._compute_optimal_M_batch(n, c, d, scalar_size=BYTES_PER_SCALAR)
        
        batches = torch.arange(n).split(M_batch_size)

        num_batches = 1 + total_points_to_sample//M_batch_size
        batches = batches[:num_batches]
        if verbose:
            print(f'Sampling AGOP on maximum of {num_batches*M_batch_size} total points')

        if verbose:
            for i, bids in tenumerate(batches):
                M.add_(self.update_M(samples[bids], p_batch_size))
        else:
            for bids in batches:
                M.add_(self.update_M(samples[bids], p_batch_size))
        
        self.M = M / (M.max() + 1e-30)
        if use_sqrtM:
            self.sqrtM = matrix_power(self.M, self.agop_power)
        del M

        
    def score(self, samples, targets, metrics):
        """
        samples: torch.Tensor of shape (n, d)
        targets: torch.Tensor of shape (n, c)
        metrics: list of metrics to compute
        """
        preds = self.predict(samples.to(self.device)).to(targets.device)
        out_metrics = {}
        if 'accuracy' in metrics:
            if preds.shape[-1]==1:
                num_classes = len(torch.unique(targets))
                if num_classes==2:
                    preds = torch.where(preds > 0.5, 1, 0).reshape(targets.shape)
                    out_metrics['accuracy'] = accuracy(preds, targets, task="binary").item()
                else:
                    out_metrics['accuracy'] = accuracy(preds, targets, task="multiclass", num_classes=num_classes).item()
            else:
                preds_ = torch.argmax(preds,dim=-1)
                targets_ = torch.argmax(targets,dim=-1)
                out_metrics['accuracy'] = accuracy(preds_, targets_, task="multiclass", num_classes=preds.shape[-1]).item()
        
        if 'mse' in metrics:
            out_metrics['mse'] = (targets - preds).pow(2).mean()
        return out_metrics

class GenericRFM(RecursiveFeatureMachine):
    """
    The preferred RFM subclass for generic kernels. These enable more fine-grained control over the kernel functions
    including adaptive bandwidths and different exponents. The kernel functions are faster and more memory efficient than
    the specific RFM subclasses.
    """
    def __init__(self, kernel: Kernel, agop_power=0.5, **kwargs):
        super().__init__(**kwargs)
        self.kernel_obj = kernel
        self.kernel_type = 'generic'
        self.agop_power = agop_power

        print("Setting kernel_obj.bandwidth to:", self.kernel_obj.bandwidth)
        print("Setting kernel_obj.exponent to:", self.kernel_obj.exponent)
        print("Setting agop_power to:", self.agop_power)
        print("Setting diag to:", self.diag)
        print("Setting centering to:", self.centering)
        
    def kernel(self, x, z):
        return self.kernel_obj.get_kernel_matrix(x, z, self.sqrtM)

    def update_M(self, samples, p_batch_size):
        samples = samples.to(self.device)
        self.centers = self.centers.to(self.device)

        if self.M is None:
            if self.diag:
                self.M = torch.ones(samples.shape[-1], device=samples.device, dtype=samples.dtype)
            else:
                self.M = torch.eye(samples.shape[-1], device=samples.device, dtype=samples.dtype)

        if self.sqrtM is None:
            if self.diag:
                self.sqrtM = torch.ones(samples.shape[-1], device=samples.device, dtype=samples.dtype)
            else:
                self.sqrtM = torch.eye(samples.shape[-1], device=samples.device, dtype=samples.dtype)

        agop_func = self.kernel_obj.get_agop_diag if self.diag else self.kernel_obj.get_agop
        agop = agop_func(x=self.centers, z=samples, coefs=self.weights.t(), mat=self.sqrtM, center_grads=self.centering)
        return agop

class LaplaceRFM(RecursiveFeatureMachine):

    def __init__(self, bandwidth=1., **kwargs):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self.kernel_type = 'laplace'

    def kernel(self, x, z):
        return laplacian_M(x, z, self.M, self.bandwidth)
    
    def update_M(self, samples, p_batch_size):
        samples = samples.to(self.device)
        self.centers = self.centers.to(self.device)
        
        # return adit_rfm.get_grads_2(self.centers, samples, self.weights.T, self.bandwidth, self.M)
        """Performs a batched update of M."""
        K = self.kernel(samples, self.centers)
        if p_batch_size is None: 
            p_batch_size = self.centers.shape[0]
        
        dist = euclidean_distances_M(samples, self.centers, self.M, squared=False)
        dist = torch.where(dist < 1e-10, torch.zeros(1, device=dist.device).float(), dist)

        K.div_(dist)
        del dist
        K[K == float("Inf")] = 0.0

        p, d = self.centers.shape
        p, c = self.weights.shape
        n, d = samples.shape

        samples_term = (K @ self.weights).reshape(n, c, 1)  # (n, p)  # (p, c)

        if self.diag:
            temp = 0
            for p_batch in torch.arange(p).split(p_batch_size):
                # temp[j,l,d] += \sum_i M[j,i] * coefs[i,l] * x[i,d]
                temp += K[:, p_batch] @ ( # (n, len(p_batch))
                    self.weights[p_batch,:].view(len(p_batch), c, 1) * (self.centers[p_batch,:] * self.M).view(len(p_batch), 1, d)
                ).reshape(
                    len(p_batch), c * d
                )  # (len(p_batch), cd)
            
            centers_term = temp.view(n, c, d)

            # M[j, i] * coefs[i, l] * z[j, d]
            samples_term = samples_term * (samples * self.M).reshape(n, 1, d)

        else:
            temp = 0
            for p_batch in torch.arange(p).split(p_batch_size):
                temp += K[:, p_batch] @ ( # (n, len(p_batch))
                    self.weights[p_batch,:].view(len(p_batch), c, 1) * (self.centers[p_batch,:] @ self.M).view(len(p_batch), 1, d)
                ).reshape(
                    len(p_batch), c * d
                )  # (len(p_batch), cd)
            
            centers_term = temp.view(n, c, d)

            samples_term = samples_term * (samples @ self.M).reshape(n, 1, d)

        G = (centers_term - samples_term) / self.bandwidth  # (n, c, d)

        del centers_term, samples_term, K
        
        if self.centering:
            G = G - G.mean(0) # (n, c, d)
        
        # return quantity to be added to M. Division by len(samples) will be done in parent function.
        if self.diag:
            return torch.einsum('ncd, ncd -> d', G, G)
        else:
            return torch.einsum("ncd, ncD -> dD", G, G)
        

class GeneralizedLaplaceRFM(RecursiveFeatureMachine):

    def __init__(self, bandwidth=1., exponent=1., agop_power=0.5, **kwargs):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self.kernel_type = 'laplacian_gen'
        self.exponent = exponent
        self.agop_power = agop_power
    
    def kernel(self, x, z):
        return laplacian_gen(x, z, self.sqrtM, self.bandwidth, self.exponent, self.diag)

    def update_M(self, samples, p_batch_size):
        
        samples = samples.to(self.device)
        self.centers = self.centers.to(self.device)
        agop = get_laplace_gen_agop(samples, 
                                    self.centers, 
                                    self.sqrtM, 
                                    self.bandwidth, 
                                    self.exponent, 
                                    self.weights, 
                                    self.diag
                                    )
        return agop

class GaussRFM(RecursiveFeatureMachine):

    def __init__(self, bandwidth=1., **kwargs):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self.kernel_type = 'gaussian'

    def kernel(self, x, z):
        return gaussian_M(x, z, self.M, self.bandwidth)

    def update_M(self, samples, p_batch_size=None):
        
        K = self.kernel(samples, self.centers)

        p, d = self.centers.shape
        p, c = self.weights.shape
        n, d = samples.shape
        
        samples_term = (
                K # (n, p)
                @ self.weights # (p, c)
            ).reshape(n, c, 1)
        
        if self.diag:
            centers_term = (
                K # (n, p)
                @ (
                    self.weights.view(p, c, 1) * (self.centers * self.M).view(p, 1, d)
                ).reshape(p, c*d) # (p, cd)
            ).view(n, c, d) # (n, c, d)

            samples_term = samples_term * (samples * self.M).reshape(n, 1, d)
            
        else:        
            centers_term = (
                K # (n, p)
                @ (
                    self.weights.view(p, c, 1) * (self.centers @ self.M).view(p, 1, d)
                ).reshape(p, c*d) # (p, cd)
            ).view(n, c, d) # (n, c, d)

            samples_term = samples_term * (samples @ self.M).reshape(n, 1, d)

        G = (centers_term - samples_term) / self.bandwidth**2 # (n, c, d)
        
        if self.centering:
            G = G - G.mean(0) # (n, c, d)
        
        if self.diag:
            return torch.einsum('ncd, ncd -> d', G, G)
        else:
            return torch.einsum("ncd, ncD -> dD", G, G)
        

class NTKModel(RecursiveFeatureMachine):
    def __init__(self, sqrtM=None, **kwargs):
        super().__init__(**kwargs)
        self.weights = None
        self.sqrtM = sqrtM

    def kernel(self, x, z):
        raise NotImplementedError("NTKModel does not implement a kernel")

    def fit(self, X, y, reg=1e-3):
        XM = X.to(self.device) @ self.sqrtM.to(X.device)
        y = y.to(self.device)
        Kmat = ntk_kernel(XM, XM)
        alphas = torch.linalg.solve(Kmat + reg * torch.eye(Kmat.shape[0], device=self.device), y)
        self.weights = alphas
        self.XM = XM.cpu()

    def predict(self, Z):
        ZM = Z.to(self.device) @ self.sqrtM.to(self.device)
        out = ntk_kernel(ZM, self.XM.to(self.device)) @ self.weights.to(self.device) # (m, c)
        return out
        

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    # define target function
    def fstar(X):
        return torch.cat([
            (X[:, 0]  > 0)[:,None],
            (X[:, 1]  < 0.1)[:,None]],
            axis=1).type(X.type())

    # create low rank data
    n = 4000
    d = 100
    torch.manual_seed(0)
    X_train = torch.randn(n,d)
    X_test = torch.randn(n,d)
    
    y_train = fstar(X_train)
    y_test = fstar(X_test)

    model = LaplaceRFM(bandwidth=1., diag=False, centering=False)
    model.fit(
        (X_train, y_train), 
        (X_test, y_test), 
        loader=False, method='eigenpro', epochs=15, print_every=5,
        iters=5,
        classif=False
    ) 
