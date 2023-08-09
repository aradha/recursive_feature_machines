try:
    from eigenpro2 import KernelModel
    EIGENPRO_AVAILABLE = True
except ModuleNotFoundError:
    print('`eigenpro2` is not installed...') 
    print('Using `torch.linalg.solve` for training the kernel model\n')
    print('WARNING: `torch.linalg.solve` scales poorly with the size of training dataset,\n '
    '         and may cause an `Out-of-Memory` error')
    print('`eigenpro2` is a more scalable solver. To use, pass `method="eigenpro"` to `model.fit()`')
    print('To install `eigenpro2` visit https://github.com/EigenPro/EigenPro-pytorch/tree/pytorch/')
    EIGENPRO_AVAILABLE = False
    
import torch, numpy as np
from torchmetrics.functional.classification import accuracy
from kernels import laplacian_M, gaussian_M, euclidean_distances_M
from tqdm.contrib import tenumerate
import hickle

class RecursiveFeatureMachine(torch.nn.Module):

    def __init__(self, device=torch.device('cpu'), mem_gb=8, diag=False, centering=False, reg=1e-3):
        super().__init__()
        self.M = None
        self.model = None
        self.diag = diag # if True, Mahalanobis matrix M will be diagonal
        self.centering = centering # if True, update_M will center the gradients before taking an outer product
        self.device = device
        self.mem_gb = mem_gb
        self.reg = reg # only used when fit using direct solve
        

    def get_data(self, data_loader):
        X, y = [], []
        for idx, batch in enumerate(data_loader):
            inputs, labels = batch
            X.append(inputs)
            y.append(labels)
        return torch.cat(X, dim=0), torch.cat(y, dim=0)

    def update_M(self):
        raise NotImplementedError("Must implement this method in a subclass")


    def fit_predictor(self, centers, targets, **kwargs):
        self.centers = centers
        if self.M is None:
            if self.diag:
                self.M = torch.ones(centers.shape[-1], device=self.device)
            else:
                self.M = torch.eye(centers.shape[-1], device=self.device)
        if self.fit_using_eigenpro and EIGENPRO_AVAILABLE:
            self.weights = self.fit_predictor_eigenpro(centers, targets, **kwargs)
        else:
            self.weights = self.fit_predictor_lstsq(centers, targets)


    def fit_predictor_lstsq(self, centers, targets):
        return torch.linalg.solve(
            self.kernel(centers, centers) 
            + self.reg*torch.eye(len(centers), device=centers.device), 
            targets
        )


    def fit_predictor_eigenpro(self, centers, targets, **kwargs):
        n_classes = 1 if targets.dim()==1 else targets.shape[-1]
        self.model = KernelModel(self.kernel, centers, n_classes, device=self.device)
        _ = self.model.fit(centers, targets, mem_gb=self.mem_gb, **kwargs)
        return self.model.weight


    def predict(self, samples):
        return self.kernel(samples, self.centers) @ self.weights


    def fit(self, train_loader, test_loader,
            iters=3, name=None, reg=1e-3, method='lstsq', 
            train_acc=False, loader=True, classif=True, **kwargs):
        # if method=='eigenpro':
        #     raise NotImplementedError(
        #         "EigenPro method is not yet supported. "+
        #         "Please try again with `method='lstlq'`")
        self.fit_using_eigenpro = (method.lower()=='eigenpro')
            # self.fit_using_eigenpro = True
        
        if loader:
            print("Loaders provided")
            X_train, y_train = self.get_data(train_loader)
            X_test, y_test = self.get_data(test_loader)
        else:
            X_train, y_train = train_loader
            X_test, y_test = test_loader

        
            
        for i in range(iters):
            self.fit_predictor(X_train, y_train, **kwargs)
            
            if classif:
                train_acc = self.score(X_train, y_train, metric='accuracy')
                print(f"Round {i}, Train Acc: {train_acc:.2f}%")
                test_acc = self.score(X_test, y_test, metric='accuracy')
                print(f"Round {i}, Test Acc: {test_acc:.2f}%")


            test_mse = self.score(X_test, y_test, metric='mse')
            print(f"Round {i}, Test MSE: {test_mse:.4f}")
            
            self.fit_M(X_train, y_train, **kwargs)

            if name is not None:
                hickle.dump(self.M, f"saved_Ms/M_{name}_{i}.h")

        self.fit_predictor(X_train, y_train, **kwargs)
        final_mse = self.score(X_test, y_test, metric='mse')
        print(f"Final MSE: {final_mse:.4f}")
        if classif:
            final_test_acc = self.score(X_test, y_test, metric='accuracy')
            print(f"Final Test Acc: {final_test_acc:.2f}%")
            
        return final_mse
    
    def fit_M(self, samples, labels, M_batch_size=None, **kwargs):
        """Applies EGOP to update the Mahalanobis matrix M."""
        
        n, d = samples.shape
        M = torch.zeros_like(self.M) if self.M is not None else (
            torch.zeros(d, dtype=samples.dtype) if self.diag else torch.zeros(d, d, dtype=samples.dtype))
        
        if M_batch_size is None: # calculate optimal batch size for batched EGOP
            curr_mem_use = torch.cuda.memory_allocated() # in bytes
            BYTES_PER_SCALAR = 8
            p, d = samples.shape
            c = labels.shape[-1]
            M_mem = (d if self.diag else d**2)
            centers_mem = (p * d)
            mem_available = (self.mem_gb *1024**3) - curr_mem_use - (M_mem + centers_mem + 2*p*d) * BYTES_PER_SCALAR
            # maximum batch size limited by K, dist, centers_term, samples_term, and G
            M_batch_size = mem_available // ((2*d + 2*p + 3*c*d + 2)*BYTES_PER_SCALAR)
        
        batches = torch.randperm(n).split(M_batch_size)
        for i, bids in tenumerate(batches):
            M += self.update_M(samples[bids])
            
        self.M = M / n
        del M

        if self.centering:
            self.M = self.M - self.M.mean(0)

    def score(self, samples, targets, metric='mse'):
        preds = self.predict(samples)
        if metric=='accuracy':
            if preds.shape[-1]==1:
                num_classes = len(torch.unique(preds))
                if num_classes==2:
                    return accuracy(preds, targets, task="binary").item()
                else:
                    return accuracy(preds, targets, task="multiclass", num_classes=num_classes).item()
            else:
                return accuracy(preds, targets, task="multilabel", num_labels=preds.shape[-1]).item()
        
        elif metric=='mse':
            return (targets - preds).pow(2).mean()


class LaplaceRFM(RecursiveFeatureMachine):

    def __init__(self, bandwidth=1., **kwargs):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self.kernel = lambda x, z: laplacian_M(x, z, self.M, self.bandwidth) # must take 3 arguments (x, z, M)
    
    def update_M(self, samples):
        """Performs a batched update of M."""
        K = self.kernel(samples, self.centers)

        dist = euclidean_distances_M(samples, self.centers, self.M, squared=False)
        dist = torch.where(dist < 1e-10, torch.zeros(1, device=dist.device).float(), dist)

        K = K / dist
        K[K == float("Inf")] = 0.0

        p, d = self.centers.shape
        p, c = self.weights.shape
        n, d = samples.shape

        samples_term = (K @ self.weights).reshape(n, c, 1)  # (n, p)  # (p, c)

        if self.diag:
            centers_term = (
                K  # (n, p)
                @ (
                    self.weights.view(p, c, 1) * (self.centers * self.M).view(p, 1, d)
                ).reshape(
                    p, c * d
                )  # (p, cd)
            ).view(
                n, c, d
            )  # (n, c, d)

            samples_term = samples_term * (samples * self.M).reshape(n, 1, d)

        else:
            centers_term = (
                K  # (n, p)
                @ (
                    self.weights.view(p, c, 1) * (self.centers @ self.M).view(p, 1, d)
                ).reshape(
                    p, c * d
                )  # (p, cd)
            ).view(
                n, c, d
            )  # (n, c, d)

            samples_term = samples_term * (samples @ self.M).reshape(n, 1, d)

        G = (centers_term - samples_term) / self.bandwidth  # (n, c, d)

        del centers_term, samples_term, K, dist
        
        # return quantity to be added to M. Division by len(samples) will be done in parent function.
        if self.diag:
            return torch.einsum('ncd, ncd -> d', G, G)
        else:
            return torch.einsum("ncd, ncD -> dD", G, G)

class GaussRFM(RecursiveFeatureMachine):

    def __init__(self, bandwidth=1., **kwargs):
        super().__init__(**kwargs)
        self.bandwidth = bandwidth
        self.kernel = lambda x, z: gaussian_M(x, z, self.M, self.bandwidth) # must take 3 arguments (x, z, M)
        

    def update_M(self, samples):
        
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
            self.M = torch.einsum('ncd, ncd -> d', G, G)/len(samples)
        else:
            self.M = torch.einsum('ncd, ncD -> dD', G, G)/len(samples)

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
