import numpy as np
import torch
from rfm import LaplaceRFM, GeneralizedLaplaceRFM
from rfm.kernels_new import LaplaceKernel, ProductLaplaceKernel
from rfm.recursive_feature_machine import GenericRFM
import time

np.random.seed(0)
torch.manual_seed(0)

M_batch_size = 2048

def fstar(X):
    return torch.cat([
            (X[:, 0]  > 0)[:,None],
    	    (X[:, 1] < 0.5)[:, None]], 
    	    axis=1
        ).float()

# model = LaplaceRFM(bandwidth=1., diag=True)
# model = GenericRFM(LaplaceKernel(bandwidth=1., exponent=1.0), diag=True)
# model = GeneralizedLaplaceRFM(bandwidth=50., exponent=1.0, diag=True)
model = GenericRFM(ProductLaplaceKernel(bandwidth=50., exponent=1.0), diag=True, bandwidth_mode='adaptive', reg=1e-3)


n = 4000 # samples
d = 100  # dimension
c = 2    # classes

X_train = torch.randn(n, d)
X_test = torch.randn(n, d)
y_train = fstar(X_train)
y_test = fstar(X_test)

start_time = time.time()

model.fit(
    (X_train, y_train), 
    (X_test, y_test), 
    loader=False, 
    iters=5,
    classif=False,
    M_batch_size=M_batch_size,
)

print(f'Time: {time.time()-start_time:g} s')
