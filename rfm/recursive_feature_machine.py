from eigenpro2 import KernelModel
import torch, numpy as np
from .kernels import laplacian_M, euclidean_distances_M
from tqdm import tqdm
import hickle

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEV_MEM_GB = torch.cuda.get_device_properties(DEVICE).total_memory//1024**3 - 1 # GPU memory in GB, keeping aside 1GB for safety
else:
    DEVICE = torch.device("cpu")
    DEV_MEM_GB = 8


class RecursiveFeatureMachine:

    def __init__(self):
        self.M = None
        self.model = None
        self.device = DEVICE
        self.mem_gb = DEV_MEM_GB

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
            self.M = torch.eye(centers.shape[-1])
        if (len(centers) > 20_000) or self.fit_using_eigenpro:
            self.weights = self.fit_predictor_eigenpro(centers, targets, **kwargs)
        else:
            self.weights = self.fit_predictor_lstsq(centers, targets)


    def fit_predictor_lstsq(self, centers, targets):
        return torch.linalg.solve(
            self.kernel(centers, centers) 
            + 1e-3*torch.eye(len(centers)), 
            targets
        )


    def fit_predictor_eigenpro(self, centers, targets, **kwargs):
        n_classes = 1 if targets.dim()==1 else targets.shape[-1]
        self.model = KernelModel(self.kernel, centers, n_classes, device=DEVICE)
        _ = self.model.fit(centers, targets, mem_gb=self.mem_gb, **kwargs)
        return self.model.weights


    def predict(self, samples):
        return self.kernel(samples, self.centers) @ self.weights


    def fit(self, train_loader, test_loader,
            iters=3, name=None, reg=1e-3, method='lstsq', 
            train_acc=False, loader=True, classif=True):
        if method=='eigenpro':
            raise NotImplementedError(
                "EigenPro method is not yet supported. "+
                "Please try again with `method='lstlq'`")
            #self.fit_using_eigenpro = (method.lower()=='eigenpro')
        self.fit_using_eigenpro = False
        
        if loader:
            print("Loaders provided")
            X_train, y_train = self.get_data(train_loader)
            X_test, y_test = self.get_data(test_loader)
        else:
            X_train, y_train = train_loader
            X_test, y_test = test_loader
            
        for i in range(iters):
            self.fit_predictor(X_train, y_train)
            
            if classif:
                train_acc = self.score(X_train, y_train, metric='accuracy')
                print(f"Round {i}, Train Acc: {train_acc:.2f}%")
                test_acc = self.score(X_train, y_train, metric='accuracy')
                print(f"Round {i}, Test Acc: {test_acc:.2f}%")


            test_mse = self.score(X_test, y_test, metric='mse')
            print(f"Round {i}, Test MSE: {test_mse:.4f}")
            
            self.update_M(X_train)

            if name is not None:
                hickle.dump(M, f"saved_Ms/M_{name}_{i}.h")

        self.fit_predictor(X_train, y_train)
        final_mse = self.score(X_test, y_test, metric='mse')
        print(f"Final MSE: {final_mse:.4f}")
        if classif:
            final_test_acc = self.score(X_test, y_test, metric='accuracy')
            print(f"Final Test Acc: {final_test_acc:.2f}%")
            
        return final_mse

    def score(self, samples, targets, metric='mse'):
        preds = self.predict(samples)
        if metric=='accuracy':
            return (1.*(targets.argmax(-1) == preds.argmax(-1))).mean()*100.
        elif metric=='mse':
            return (targets - preds).pow(2).mean()


class LaplaceRFM(RecursiveFeatureMachine):

    def __init__(self, bandwidth=1.):
        super(LaplaceRFM, self).__init__()
        self.bandwidth = bandwidth
        self.kernel = lambda x, z: laplacian_M(x, z, self.M, self.bandwidth) # must take 3 arguments (x, z, M)
        

    def update_M(self, samples):
        
        K = self.kernel(samples, self.centers)

        dist = euclidean_distances_M(samples, self.centers, self.M, squared=False)
        dist = torch.where(dist < 1e-10, torch.zeros(1).float(), dist)

        K = K/dist
        K[K == float("Inf")] = 0.

        p, d = self.centers.shape
        p, c = self.weights.shape
        n, d = samples.shape
        
        centers_term = (
            K # (n, p)
            @ (
                self.weights.view(p, c, 1) * (self.centers @ self.M).view(p, 1, d)
            ).reshape(p, c*d) # (p, cd)
        ).view(n, c, d)

        samples_term = (
                K # (n, p)
                @ self.weights # (p, c)
            ).reshape(n, c, 1)
        samples_term = samples_term * (samples @ self.M).reshape(n, 1, d)

        G = (centers_term - samples_term) / self.bandwidth
        self.M = torch.einsum('ncd, ncD -> dD', G, G)/len(samples)



if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    
    # define target function
    def fstar(X):
        return torch.cat([
            (X[:, 0]  > 0)[:,None],
            (X[:, 1]  < 0.1)[:,None]],
            axis=1)


    # create low rank data
    n = 4000
    d = 100
    np.random.seed(0)
    X_train = torch.from_numpy(np.random.normal(scale=0.5, size=(n,d)))
    X_test = torch.from_numpy(np.random.normal(scale=0.5, size=(n,d)))

    y_train = fstar(X_train).double()
    y_test = fstar(X_test).double()

    model = LaplaceRFM(bandwidth=1.)
    model.fit(
        (X_train, y_train), 
        (X_test, y_test), 
        loader=False,
        iters=5,
        classif=False
    ) 
