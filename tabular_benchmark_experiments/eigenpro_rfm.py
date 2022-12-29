import torch
from torch.autograd import Variable
import torch.optim as optim
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import classic_kernel
from numpy.linalg import svd, solve, norm
import hickle
from tqdm import tqdm
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from sklearn.svm import SVC
import eigenpro


def laplace_kernel_M(pair1, pair2, bandwidth, M):
    return classic_kernel.laplacian_M(pair1, pair2, bandwidth, M)

def laplace_kernel(pair1, pair2, bandwidth):
    return classic_kernel.laplacian(pair1, pair2, bandwidth)

def kernel(pair1, pair2, nngp=False):
    out = pair1 @ pair2.transpose(1, 0)
    N1 = torch.sum(torch.pow(pair1, 2), dim=-1).view(-1, 1)
    N2 = torch.sum(torch.pow(pair2, 2), dim=-1).view(-1, 1)

    XX = torch.sqrt(N1 @ N2.transpose(1, 0))
    out = out / XX

    out = torch.clamp(out, -1, 1)

    first = 1/np.pi * (out * (np.pi - torch.acos(out)) \
                       + torch.sqrt(1. - torch.pow(out, 2))) * XX
    if nngp:
        out = first
    else:
        sec = 1/np.pi * out * (np.pi - torch.acos(out)) * XX
        out = first + sec

    return out


def get_grads(X, sol, L, P):
    M = 0.

    start = time.time()
    num_samples = 20000
    indices = np.random.randint(len(X), size=num_samples)

    if len(X) > len(indices):
        x = X[indices, :]
    else:
        x = X

    K = laplace_kernel_M(X, x, L, P)

    dist = classic_kernel.euclidean_distances_M(X, x, P, squared=False)
    dist = torch.where(dist < 1e-10, torch.zeros(1).float(), dist)

    K = K/dist
    K[K == float("Inf")] = 0.

    a1 = torch.from_numpy(sol.T).float()
    n, d = X.shape
    n, c = a1.shape
    m, d = x.shape

    a1 = a1.reshape(n, c, 1)
    X1 = (X @ P).reshape(n, 1, d)
    step1 = a1 @ X1
    del a1, X1
    step1 = step1.reshape(-1, c*d)

    step2 = K.T @ step1
    del step1

    step2 = step2.reshape(-1, c, d)

    a2 = torch.from_numpy(sol).float()
    step3 = (a2 @ K).T

    del K, a2

    step3 = step3.reshape(m, c, 1)
    x1 = (x @ P).reshape(m, 1, d)
    step3 = step3 @ x1

    G = (step2 - step3) * -1/L

    M = 0.

    bs = 50
    batches = torch.split(G, bs)
    for i in tqdm(range(len(batches))):
    #for i in range(len(batches)):
        grad = batches[i].cuda()
        gradT = torch.transpose(grad, 1, 2)
        M += torch.sum(gradT @ grad, dim=0).cpu()
        del grad, gradT
    torch.cuda.empty_cache()
    M /= len(G)
    M = M.numpy()

    return M


def convert_one_hot(y, c):
    o = np.zeros((y.size, c))
    o[np.arange(y.size), y] = 1
    return o


def eigenpro_solve(X_train, y_train, X_test, y_test, c, L, steps, M):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    M = M.cuda()
    kernel_fn = lambda x, y: laplace_kernel_M(x, y, L, M)
    model = eigenpro.FKR_EigenPro(kernel_fn, X_train, c, device=device)

    res = model.fit(X_train, y_train, X_test, y_test,
                    epochs=list(range(steps)), mem_gb=12,
                    n_subsamples=2000)
    best_acc = 0
    best_ep_iter = 0
    for idx, r in enumerate(res):
        acc = res[r][-2]['multiclass-acc']
        if acc > best_acc:
            best_ep_iter = idx
            best_acc = acc

    return model.weight.cpu().numpy(), best_acc, best_ep_iter, acc


def hyperparam_train(X_train, y_train, X_test, y_test, c,
                     iters=5, ep_iter=10, L=10, normalize=False):

    y_train = convert_one_hot(y_train, c)
    y_test = convert_one_hot(y_test, c)

    if normalize:
        X_train /= norm(X_train, axis=-1).reshape(-1, 1)
        X_test /= norm(X_test, axis=-1).reshape(-1, 1)

    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')

    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')

    best_acc = 0.
    best_iter = 0.
    best_ep_iter = 0

    n, d = X_train.shape
    M = np.eye(d, dtype='float32')
    best_M = M

    for i in range(iters):
        sol, old_test_acc, s_ep_iter, _ = eigenpro_solve(X_train, y_train,
                                                         X_test, y_test,
                                                         c, L, ep_iter,
                                                         torch.from_numpy(M))

        sol = sol.T
        if old_test_acc >= best_acc:
            best_iter = i
            best_acc = old_test_acc
            best_M = M
            best_ep_iter = s_ep_iter
        M  = get_grads(torch.from_numpy(X_train).float(), sol,
                       L, torch.from_numpy(M))

    return best_acc, best_iter, best_M, best_ep_iter


def train(X_train, y_train, X_test, y_test, c, M,
          iters=5, ep_iter=0, L=10, normalize=False):

    y_train = convert_one_hot(y_train, c)
    y_test = convert_one_hot(y_test, c)

    if normalize:
        X_train /= norm(X_train, axis=-1).reshape(-1, 1)
        X_test /= norm(X_test, axis=-1).reshape(-1, 1)

    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')

    sol, best_acc, _, acc = eigenpro_solve(X_train, y_train,
                                           X_test, y_test,
                                           c, L, ep_iter,
                                           torch.from_numpy(M))
    return acc
