import numpy as np
import torch
from rfm import LaplaceRFM
from rfm.kernels_new import LaplaceKernel
from rfm.recursive_feature_machine import GenericRFM

np.random.seed(0)
torch.manual_seed(0)

def fstar(X):
    return torch.cat([
            (X[:, 0]  > 0)[:,None],
    	    (X[:, 1] < 0.5)[:, None]], 
    	    axis=1
        ).float()

model = LaplaceRFM(bandwidth=1., diag=False)
# model = GenericRFM(LaplaceKernel(bandwidth=1., exponent=0.8), diag=True)

n = 4000 # samples
d = 4  # dimension
c = 2    # classes

X_train = torch.randn(n, d)
X_test = torch.randn(n, d)
y_train = fstar(X_train)
y_test = fstar(X_test)

model.fit(
    (X_train, y_train), 
    (X_test, y_test), 
    loader=False, 
    iters=5,
    classif=False
)
