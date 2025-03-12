import numpy as np
import torch
from rfm import LaplaceRFM, GeneralizedLaplaceRFM
from rfm.kernels_new import LaplaceKernel, ProductLaplaceKernel
from rfm.recursive_feature_machine import GenericRFM

np.random.seed(0)
torch.manual_seed(0)

def fstar(X):
    return torch.cat([
            (X[:, 0]  > 0)[:,None],
    	    (X[:, 1] < 0.5)[:, None]], 
    	    axis=1
        ).float()

# model = LaplaceRFM(bandwidth=1., diag=True)
# model = GenericRFM(LaplaceKernel(bandwidth=1., exponent=1.0), diag=True)
# model = GeneralizedLaplaceRFM(bandwidth=50., exponent=1.0, diag=True)
model = GenericRFM(ProductLaplaceKernel(bandwidth=10., exponent=1.0), diag=True)


n = 1000 # samples
d = 30  # dimension
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
