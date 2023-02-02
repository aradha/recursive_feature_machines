# Recursive Feature Machines

There are two notebooks to test out RFM: 
- low_rank.ipynb (an example of low rank polynomials)
- svhn.ipynb (for the SVHN dataset)

## Requirements:
1. Python 3
2. pytorch>=1.13
3. torchvision==0.14.0
4. hickle==5.0.2
5. tqdm
6. eigenpro>=2.0 \
`eigenpro` can be installed from [https://github.com/EigenPro/EigenPro-pytorch/tree/pytorch](github.com/EigenPro/EigenPro-pytorch)

Can be installed using the command
```
 pip install git+https://github.com/aradha/recursive_feature_machines.git@pip_install
```

## Stable behavior
Code has been tested using PyTorch 1.13, Python 3.8

## Testing installation
```python
from rfm import RecursiveFeatureMachine
from rfm.kernels import laplacian_M, laplacian_M_grad1
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
```
