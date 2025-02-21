from .eigenpro import KernelModel
    
import torch, numpy as np
from torchmetrics.functional.classification import accuracy
from .kernels import laplacian_M, gaussian_M, euclidean_distances_M, laplacian_gen, get_laplace_gen_agop, ntk_kernel
from tqdm.contrib import tenumerate
import hickle
from .utils import matrix_sqrt
from time import time

class RecursiveFeatureMachine(torch.nn.Module):

    def __init__(self, device=torch.device('cpu'), mem_gb=8, diag=False, centering=False, reg=1e-3, iters=5, p_batch_size=None):
        super().__init__()
        self.M = None
        self.sqrtM = None
        self.model = None
        self.diag = diag # if True, Mahalanobis matrix M will be diagonal
        self.centering = centering # if True, update_M will center the gradients before taking an outer product
        self.device = device
        self.mem_gb = mem_gb
        self.reg = reg # only used when fit using direct solve
        self.iters = iters
        self.kernel_type = None
        self.p_batch_size = p_batch_size
        

    def get_data(self, data_loader):
        X, y = [], []
        for idx, batch in enumerate(data_loader):
            inputs, labels = batch
            X.append(inputs)
            y.append(labels)
        return torch.cat(X, dim=0), torch.cat(y, dim=0)

    def update_M(self):
        raise NotImplementedError("Must implement this method in a subclass")


    def fit_predictor(self, centers, targets, classification=False, class_weight=None, bs=None, lr_scale=1, **kwargs):
        self.centers = centers
        if self.M is None:
            if self.diag:
                self.M = torch.ones(centers.shape[-1], device=self.device, dtype=centers.dtype)
            else:
                self.M = torch.eye(centers.shape[-1], device=self.device, dtype=centers.dtype)
        if self.fit_using_eigenpro:
            self.weights = self.fit_predictor_eigenpro(centers, targets, bs=bs, lr_scale=lr_scale, 
                                                       classification=classification, **kwargs)
        else:
            self.weights = self.fit_predictor_lstsq(centers, targets, class_weight=class_weight)


    def fit_predictor_lstsq(self, centers, targets, class_weight=None):
        centers = centers.to(self.device)
        targets = targets.to(self.device)

        assert(len(centers)==len(targets))

        kernel_matrix = self.kernel(centers, centers)    

        if class_weight == 'inverse':
            flat_targets = targets.flatten().long()
            class_weights = {
                0: len(flat_targets) / (2 * (flat_targets == 0).sum()),
                1: len(flat_targets) / (2 * (flat_targets == 1).sum())
            }

            sample_weights = torch.ones_like(targets, dtype=kernel_matrix.dtype).flatten()
            sample_weights[flat_targets==0] = class_weights[0]
            sample_weights[flat_targets==1] = class_weights[1]
            sample_weights = sample_weights.to(device=self.device, dtype=kernel_matrix.dtype)
            
            W = torch.diag(sample_weights)
            kernel_matrix = W@kernel_matrix
            targets = W@targets

        if self.reg > 0:
            kernel_matrix += self.reg * torch.eye(len(centers), device=self.device)
            
        return torch.linalg.solve(
            kernel_matrix, 
            targets
        )


    def fit_predictor_eigenpro(self, centers, targets, bs, lr_scale, **kwargs):
        n_classes = 1 if targets.dim()==1 else targets.shape[-1]
        self.model = KernelModel(self.kernel, centers, n_classes, device=self.device)
        _ = self.model.fit(centers, targets, mem_gb=self.mem_gb, bs=bs, lr_scale=lr_scale, **kwargs)
        return self.model.weight


    def predict(self, samples):
        out = self.kernel(samples.to(self.device), self.centers.to(self.device)) @ self.weights.to(self.device)
        return out.to(samples.device)


    def fit(self, train_loader, test_loader,
            iters=None, name=None, method='lstsq', 
            train_acc=False, loader=True, classification=True, 
            return_mse=False, verbose=True, M_batch_size=None, 
            class_weight=None, return_best_params=False, bs=None, lr_scale=1,
            total_points_to_sample=50000, **kwargs):
                
        self.fit_using_eigenpro = (method.lower()=='eigenpro')
        use_sqrtM = self.kernel_type in ['laplacian_gen']
        
        if iters is None:
            iters = self.iters

        if class_weight is not None and self.fit_using_eigenpro:
            raise ValueError("Class weights are not supported for EigenPro")

        if verbose and class_weight == 'inverse':
            print("Weighting samples by inverse class frequency")
        
        if loader:
            print("Loaders provided")
            X_train, y_train = self.get_data(train_loader)
            X_test, y_test = self.get_data(test_loader)
        else:
            X_train, y_train = train_loader
            X_test, y_test = test_loader

        
        mses, Ms = [], []
        best_alphas, best_M, best_sqrtM = None, None, None
        best_metric = float('inf') if not classification else 0 
        for i in range(iters):
            self.fit_predictor(X_train, y_train, X_val=X_test, y_val=y_test, 
                               classification=classification, class_weight=class_weight, 
                               bs=bs, lr_scale=lr_scale, **kwargs)
            
            if classification:
                test_acc = self.score(X_test, y_test, bs, metric='accuracy')
                if method == 'lstsq':
                    train_acc = self.score(X_train, y_train, bs, metric='accuracy')
                    if verbose:
                        print(f"Round {i}, Train Acc: {100*train_acc:.2f}%, Test Acc: {100*test_acc:.2f}%")
                else:
                    if verbose:
                        print(f"Round {i}, Test Acc: {100*test_acc:.2f}%")

            test_mse = self.score(X_test, y_test, bs, metric='mse')

            if verbose:
                print(f"Round {i}, Test MSE: {test_mse:.4f}")

            # if classification and accuracy higher, or if regression and mse lower
            if return_best_params and classification and test_acc > best_metric:
                best_metric = test_acc
                best_alphas = self.weights.cpu().clone()
                best_M = self.M.cpu().clone()
                if use_sqrtM:
                    best_sqrtM = matrix_sqrt(self.M).cpu().clone()
            elif return_best_params and not classification and test_mse < best_metric:
                best_metric = test_mse
                best_alphas = self.weights.cpu().clone()
                best_M = self.M.cpu().clone()
                if use_sqrtM:
                    best_sqrtM = matrix_sqrt(self.M).cpu().clone()

            self.fit_M(X_train, y_train, verbose=verbose, M_batch_size=M_batch_size, use_sqrtM=use_sqrtM, total_points_to_sample=total_points_to_sample, **kwargs)
   
            
            if return_mse:
                Ms.append(self.M+0)
                mses.append(test_mse)

            if name is not None:
                hickle.dump(self.M, f"saved_Ms/M_{name}_{i}.h")

        self.fit_predictor(X_train, y_train, X_val=X_test, y_val=y_test, class_weight=class_weight, classification=classification, bs=bs, **kwargs)
        final_mse = self.score(X_test, y_test, bs=bs, metric='mse')
        
        if verbose:
            print(f"Final MSE: {final_mse:.4f}")
        if classification:
            final_test_acc = self.score(X_test, y_test, bs=bs, metric='accuracy')
            if verbose:
                print(f"Final Test Acc: {100*final_test_acc:.2f}%")

        # if classification and accuracy higher, or if regression and mse lower
        if return_best_params and classification and final_test_acc > best_metric:
            best_metric = final_test_acc
            best_alphas = self.weights.cpu().clone()
            best_M = self.M.cpu().clone()
            if use_sqrtM:
                best_sqrtM = matrix_sqrt(self.M).cpu().clone()
        elif return_best_params and not classification and final_mse < best_metric:
            best_metric = final_mse
            best_alphas = self.weights.cpu().clone()
            best_M = self.M.cpu().clone()
            if use_sqrtM:
                best_sqrtM = matrix_sqrt(self.M).cpu().clone()

        if return_best_params:
            print(f"Returning best parameters with value: {best_metric}")
            self.M = best_M.to(self.device)
            self.weights = best_alphas.to(self.device)
            if use_sqrtM:
                self.sqrtM = best_sqrtM.to(self.device)

        if return_mse:
            return Ms, mses
            
        return final_mse
    
    def _compute_optimal_M_batch(self, p, c, d, scalar_size=4):
        """Computes the optimal batch size for EGOP."""
        THREADS_PER_BLOCK = 512 # pytorch default
        def tensor_mem_usage(numels):
            """Calculates memory footprint of tensor based on number of elements."""
            return np.ceil(scalar_size * numels / THREADS_PER_BLOCK) * THREADS_PER_BLOCK

        def max_tensor_size(mem):
            """Calculates maximum possible tensor given memory budget (bytes)."""
            return int(np.floor(mem / THREADS_PER_BLOCK) * (THREADS_PER_BLOCK / scalar_size))

        curr_mem_use = torch.cuda.memory_allocated() # in bytes
        M_mem = tensor_mem_usage(d if self.diag else d**2)
        centers_mem = tensor_mem_usage(p * d)
        mem_available = (self.mem_gb *1024**3) - curr_mem_use - (M_mem + centers_mem) * scalar_size
        M_batch_size = max_tensor_size((mem_available - 3*tensor_mem_usage(p) - tensor_mem_usage(p*c*d)) / (2*scalar_size*(1+p)))

        return M_batch_size
    
    def fit_M(self, samples, labels, p_batch_size=None, M_batch_size=None, 
              verbose=True, total_points_to_sample=50000, use_sqrtM=False, **kwargs):
        """Applies EGOP to update the Mahalanobis matrix M."""
        
        n, d = samples.shape
        M = torch.zeros_like(self.M) if self.M is not None else (
            torch.zeros(d, dtype=samples.dtype) if self.diag else torch.zeros(d, d, dtype=samples.dtype))
        
        if M_batch_size is None: 
            BYTES_PER_SCALAR = self.M.element_size()
            p, d = samples.shape
            c = labels.shape[-1]
            M_batch_size = self._compute_optimal_M_batch(p, c, d, scalar_size=BYTES_PER_SCALAR)

            if verbose:
                print(f"Using batch size of {M_batch_size}")
        
        batches = torch.randperm(n).split(M_batch_size)

        num_batches = 1 + total_points_to_sample//M_batch_size
        batches = batches[:num_batches]
        if verbose:
            print(f'Sampling AGOP on {num_batches*M_batch_size} total points')

        if verbose:
            for i, bids in tenumerate(batches):
                torch.cuda.empty_cache()
                M.add_(self.update_M(samples[bids], p_batch_size))
        else:
            for bids in batches:
                torch.cuda.empty_cache()
                M.add_(self.update_M(samples[bids], p_batch_size))
            
        self.M = M / M.max()
        if use_sqrtM:
            self.sqrtM = matrix_sqrt(self.M)
        del M

        
    def score(self, samples, targets, bs, metric='mse'):
        if bs is None:
            preds = self.predict(samples.to(self.device)).to(targets.device)
        else:
            all_preds = []
            for batch in samples.split(bs):
                preds = self.predict(batch.to(self.device)).to(targets.device)
                all_preds.append(preds)
            preds = torch.cat(all_preds, dim=0).to(targets.device)
        if metric=='accuracy':
            if preds.shape[-1]==1:
                num_classes = len(torch.unique(targets))
                if num_classes==2:
                    preds = torch.where(preds > 0.5, 1, 0).reshape(targets.shape)
                    return accuracy(preds, targets, task="binary").item()
                else:
                    return accuracy(preds, targets, task="multiclass", num_classes=num_classes).item()
            else:
                preds_ = torch.argmax(preds,dim=-1)
                targets_ = torch.argmax(targets,dim=-1)
                return accuracy(preds_, targets_, task="multiclass", num_classes=preds.shape[-1]).item()
        
        elif metric=='mse':
            return (targets - preds).pow(2).mean()


class LaplaceRFM(RecursiveFeatureMachine):

    def __init__(self, bandwidth=1., **kwargs):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self.kernel = lambda x, z: laplacian_M(x, z, self.M, self.bandwidth) # must take 3 arguments (x, z, M)
        self.kernel_type = 'laplace'
    
    def update_M(self, samples, p_batch_size):
        samples = samples.to(self.device)
        self.centers = self.centers.to(self.device)
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
                temp += K[:, p_batch] @ ( # (n, len(p_batch))
                    self.weights[p_batch,:].view(len(p_batch), c, 1) * (self.centers[p_batch,:] * self.M).view(len(p_batch), 1, d)
                ).reshape(
                    len(p_batch), c * d
                )  # (len(p_batch), cd)
            
            centers_term = temp.view(n, c, d)

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

    def __init__(self, bandwidth=1., exponent=1., **kwargs):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self.kernel = lambda x, z: laplacian_gen(x, z,  self.sqrtM, self.bandwidth, exponent)
        self.kernel_type = 'laplacian_gen'
        self.exponent = exponent
        

    def update_M(self, samples, p_batch_size):
        samples_batch_size = self.p_batch_size
        
        if self.M is None:
            self.M = torch.eye(samples.shape[-1], device=samples.device, dtype=samples.dtype)
            self.sqrtM = torch.eye(samples.shape[-1], device=samples.device, dtype=samples.dtype)

        samples = samples.to(self.device)
        self.centers = self.centers.to(self.device)
        agop = get_laplace_gen_agop(samples, self.centers, self.sqrtM, self.bandwidth, self.exponent, self.weights, samples_batch_size)
        return agop


class GaussRFM(RecursiveFeatureMachine):

    def __init__(self, bandwidth=1., **kwargs):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self.kernel = lambda x, z: gaussian_M(x, z, self.M, self.bandwidth) # must take 3 arguments (x, z, M)
        self.kernel_type = 'gaussian'

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
