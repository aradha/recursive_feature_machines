import torch
from rfm import LaplaceRFM

def fstar(X):
    return torch.cat([
            (X[:, 0]  > 0)[:,None], 
    	    (X[:, 1] < 0.5)[:, None]], 
    	    axis=1
        ).float()

model =LaplaceRFM(bandwidth=1.)

n = 4000 # samples
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
