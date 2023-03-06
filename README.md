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

## Stable behavior
Code has been tested using PyTorch 1.13, Python 3.8

## Testing installation
```python
import torch
from rfm import LaplaceRFM

def fstar(X):
    return torch.cat([
        (X[:, 0]  > 0)[:,None], 
	(X[:, 1] < 0.5)[:, None]], 
	axis=1).float()

model =LaplaceRFM(bandwidth=1.)

n = 1000 # samples
d = 100  # dimension
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
```


# Paper
[Feature learning in neural networks and kernel machines that recursively learn features](https://arxiv.org/abs/2212.13881)
