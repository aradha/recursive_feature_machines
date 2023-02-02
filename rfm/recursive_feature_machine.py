import numpy as np
from eigenpro import EigenProRegressor
import torch
from numpy.linalg import solve
from tqdm import tqdm
import hickle

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEV_MEM_GB = torch.cuda.get_device_properties(DEVICE).total_memory//1024**3 - 1 # GPU memory in GB, keeping aside 1GB for safety
else:
    DEVICE = torch.device("cpu")
    DEV_MEM_GB = 8


class RecursiveFeatureMachine:

    def __init__(self, kernel_fn, kernel_grad1, per_class=False):
        self.M = None
        self.per_class_str = 'c' if per_class else ''
        self.kernel = lambda x, z: kernel_fn(x, z, self.M) # must take 3 arguments (x, z, M)
        self.kernel_grad1 = lambda x, z: kernel_grad1(x, z, self.M)
        self.model = None
        self.device = DEVICE
        self.mem_gb = DEV_MEM_GB


    def get_data(self, data_loader):
        X = []
        y = []
        for idx, batch in enumerate(data_loader):
            inputs, labels = batch
            X.append(inputs)
            y.append(labels)
        return torch.cat(X, dim=0), torch.cat(y, dim=0)

    def reduce(self, samples, centers, Ksc, alpha):

        M1 = torch.einsum(
            f"md, mn, nc, mD, mn, nc -> {self.per_class_str}dD", 
            samples, Ksc, alpha, 
            samples, Ksc, alpha
            )
        M2 = torch.einsum(
            f"nd, mn, nc, nD, mn, nc -> {self.per_class_str}dD", 
            centers, Ksc, alpha,
            centers, Ksc, alpha
        )
        M12 = torch.einsum(
            f"md, mn, nc, nD, mn, nc -> {self.per_class_str}dD", 
            samples, Ksc, alpha,
            centers, Ksc, alpha
        )
        return (M1 + M2 - M12 - M12.swapaxes(-1, -2))/samples.shape[0]

    def update_M(self):
        num_samples = 20000
        indices = torch.randperm(len(self.centers))[:num_samples]
        samples = self.centers[indices]

        K = self.kernel_grad1(samples, self.centers)

        self.M = self.reduce(samples, self.centers, K, self.weights)
        

    def fit_predictor(self, centers, targets, **kwargs):
        self.centers = centers
        if self.M is None:
            self.M = torch.eye(centers.shape[-1])
        if (len(X_train) > 20_000):
            self.weights = self.fit_predictor_eigenpro(centers, targets, **kwargs)
        else:
            self.weights = self.fit_predictor_lstsq(centers, targets)


    def fit_predictor_lstsq(self, centers, targets):
        return torch.linalg.solve(self.kernel(centers, centers), targets)


    def fit_predictor_eigenpro(self, centers, targets, **kwargs):
        n_classes = 1 if targets.dim()==1 else targets.shape[-1]
        self.model = EigenProRegressor(self.kernel, centers, n_classes, device=DEVICE)
        _ = self.model.fit(centers, targets, mem_gb=self.mem_gb, **kwargs)
        return self.model.weights


    def predict(self, samples):
        return self.kernel(samples, self.centers) @ self.weights


    def fit(self, train_loader, test_loader,
            iters=3, name=None, reg=1e-3, eigenpro=False, 
            train_acc=False, loader=True, classif=True):
        
        if loader:
            print("Loaders provided")
            X_train, y_train = self.get_data(train_loader)
            X_test, y_test = self.get_data(test_loader)
        else:
            X_train, y_train = train_loader
            X_test, y_test = test_loader
            
            X_train = torch.from_numpy(X_train).float()
            X_test = torch.from_numpy(X_test).float()
            y_train = torch.from_numpy(y_train).float()
            y_test = torch.from_numpy(y_test).float()
            
        
        for i in range(iters):
            self.fit_predictor(X_train, y_train)
            
            if classif:
                train_acc = self.score(X_train, y_train, metric='accuracy')
                print(f"Round {i}, Train Acc: {train_acc:.2f}%")
                test_acc = self.score(X_train, y_train, metric='accuracy')
                print(f"Round {i}, Test Acc: {test_acc:.2f}%")


            test_mse = self.score(X_test, y_test, metric='mse')
            print(f"Round {i}, Test MSE: {test_mse:.2f}")
            

            self.update_M()

            if name is not None:
                hickle.dump(M, f"saved_Ms/M_{name}_{i}.h")

        self.fit_predictor(X_train, y_train)
        final_mse = self.score(X_test, y_test, metric='mse')
        print(f"Final MSE: {final_mse:.2f}")
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



if __name__ == "__main__":

    from kernels import laplacian_M, laplacian_M_grad1
    bw = 10
    kernel_fn = lambda x, z, M: laplacian_M(x, z, M, bw)
    kernel_grad1 = lambda x, z, M: laplacian_M_grad1(x, z, M, bw)
    model = RecursiveFeatureMachine(kernel_fn, kernel_grad1)
    
    n = 1000 # samples
    d = 100  # dimension
    c = 2    # classes
    
    X_train = np.random.randn(n, d)
    X_test = np.random.randn(n, d)
    y_train = np.random.randn(n, c)
    y_test = np.random.randn(n, c)

    model.fit((X_train, y_train), (X_test, y_test), loader=False)