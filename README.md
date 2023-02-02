# Recursive Feature Machines

There are two notebooks to test out RFM: 
- low_rank.ipynb (an example of low rank polynomials)
- svhn.ipynb (for the SVHN dataset)


# Installation

Can be installed using the command
```
 pip install git+https://github.com/aradha/recursive_feature_machines.git@pip_install
```
## Requirements:
- Python 3.8+
- torchvision==0.14.0
- hickle==5.0.2
- tqdm
- eigenpro>=2.0 

`eigenpro` can be installed from [github.com/EigenPro/EigenPro-pytorch](https://github.com/EigenPro/EigenPro-pytorch/tree/pytorch)


## Stable behavior
Code has been tested using PyTorch 1.13, Python 3.8

## Testing installation
```python
import torch
from rfm import RecursiveFeatureMachine
from rfm.kernels import laplacian_M, laplacian_M_grad1
bw = 10
kernel_fn = lambda x, z, M: laplacian_M(x, z, M, bw)
kernel_grad1 = lambda x, z, M: laplacian_M_grad1(x, z, M, bw)

model = RecursiveFeatureMachine(kernel_fn, kernel_grad1)

n = 1000 # samples
d = 100  # dimension
c = 2    # classes

X_train = torch.randn(n, d)
X_test = torch.randn(n, d)
y_train = torch.randn(n, c)
y_test = torch.randn(n, c)

model.fit((X_train, y_train), (X_test, y_test), loader=False)
```
