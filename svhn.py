import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from rfm import LaplaceRFM, GeneralizedLaplaceRFM, GenericRFM
from rfm.generic_kernels import LaplaceKernel, ProductLaplaceKernel
import gc

def pre_process(torchset, n_samples, num_classes=10):
    n_samples = min(n_samples, len(torchset))
    indices = list(np.random.choice(len(torchset), n_samples, replace=False))

    trainset = []
    for ix in indices:
        x,y = torchset[ix]
        ohe_y = torch.zeros(num_classes)
        ohe_y[y] = 1
        trainset.append(((x/np.linalg.norm(x)).reshape(-1),ohe_y))
    return trainset

# load svhn data
transform = transforms.Compose([
    transforms.ToTensor()
])

data_path = '../data/'
trainset0 = torchvision.datasets.SVHN(root=data_path,
                                    split = "train",
                                    transform=transform,
                                    download=True)
testset0 = torchvision.datasets.SVHN(root=data_path,
                                    split = "test",
                                    transform=transform,
                                    download=True)

trainset = pre_process(trainset0, n_samples=70000, num_classes=10)
testset = pre_process(testset0, n_samples=1000, num_classes=10)

X_train = []
y_train = []
for x, y in trainset:
    X_train.append(x)
    y_train.append(y)
X_train = torch.stack(X_train)
y_train = torch.stack(y_train)

X_test = []
y_test = []
for x, y in testset:
    X_test.append(x)
    y_test.append(y)
X_test = torch.stack(X_test)
y_test = torch.stack(y_test)

print('train', X_train.shape, y_train.shape)
print('test', X_test.shape, y_test.shape)

print("Running GenericRFM-Laplace")
model = GenericRFM(kernel=LaplaceKernel(bandwidth=10, exponent=1.), device='cuda', reg=1e-3, iters=3, bandwidth_mode='constant')  
# model = GeneralizedLaplaceRFM(device='cuda', reg=1e-5, bandwidth=10, iters=3, diag=True)   
model.fit((X_train, y_train), 
          (X_test, y_test), 
          classification=True, 
          M_batch_size=10000, 
          epochs=1,
          solver='solve',
          prefit_eigenpro=True,
          method='lstsq',
          return_best_params=True)

y_train_pred = model.predict(X_train).argmax(dim=1)
y_test_pred = model.predict(X_test).argmax(dim=1)

print(y_train_pred.shape, y_test_pred.shape)
train_acc = (y_train_pred == y_train.argmax(dim=1)).float().mean()
test_acc = (y_test_pred == y_test.argmax(dim=1)).float().mean()
print(f"train acc: {train_acc}, test acc: {test_acc}")
