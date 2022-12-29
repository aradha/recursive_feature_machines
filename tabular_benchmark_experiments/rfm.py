import numpy as np
import torch
from numpy.linalg import solve, svd, norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import classic_kernel

import time
from tqdm import tqdm
import hickle


def get_mse(y_pred, y_true):
    return np.mean(np.square(y_pred - y_true))


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

def laplace_kernel(pair1, pair2, bandwidth):
    return classic_kernel.laplacian(pair1, pair2, bandwidth)

def laplace_kernel_M(pair1, pair2, bandwidth, M):
    return classic_kernel.laplacian_M(pair1, pair2, bandwidth, M)


def original_ntk(X_train, y_train, X_test, y_test, use_nngp=False):
    K_train = kernel(X_train, X_train, nngp=use_nngp).numpy()
    sol = solve(K_train, y_train).T
    K_test = kernel(X_train, X_test, nngp=use_nngp).numpy()
    y_pred = (sol @ K_test).T

    mse = get_mse(y_pred, y_test.numpy())
    if use_nngp:
        print("Original NNGP MSE: ", mse)
        return mse
    else:
        print("Original NTK MSE: ", mse)
        return mse


def get_grads(X, sol, L, P):
    M = 0.

    start = time.time()
    num_samples = 20000
    indices = np.random.randint(len(X), size=num_samples)

    #"""
    if len(X) > len(indices):
        x = X[indices, :]
    else:
        x = X

    #n, d = X.shape
    #x = np.random.normal(size=(1000, d))
    #x = torch.from_numpy(x)

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

    bs = 10
    batches = torch.split(G, bs)
    #for i in tqdm(range(len(batches))):
    for i in range(len(batches)):
        grad = batches[i].cuda()
        gradT = torch.transpose(grad, 1, 2)
        #gradT = torch.swapaxes(grad, 1, 2)#.cuda()
        M += torch.sum(gradT @ grad, dim=0).cpu()
        del grad, gradT
    torch.cuda.empty_cache()
    M /= len(G)

    M = M.numpy()

    end = time.time()

    #print("Time: ", end - start)
    return M


def convert_one_hot(y, c):
    o = np.zeros((y.size, c))
    o[np.arange(y.size), y] = 1
    return o


def hyperparam_train(X_train, y_train, X_test, y_test, c,
                     iters=5, reg=0, L=10, normalize=False):

    y_t_orig = y_train
    y_v_orig = y_test
    y_train = convert_one_hot(y_train, c)
    y_test = convert_one_hot(y_test, c)

    if normalize:
        X_train /= norm(X_train, axis=-1).reshape(-1, 1)
        X_test /= norm(X_test, axis=-1).reshape(-1, 1)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    best_acc = 0.
    best_iter = 0.
    best_M = 0.

    n, d = X_train.shape
    M = np.eye(d, dtype='float32')

    for i in range(iters):

        K_train = laplace_kernel_M(X_train, X_train, L, torch.from_numpy(M)).numpy()
        sol = solve(K_train + reg * np.eye(len(K_train)), y_train).T

        K_test = laplace_kernel_M(X_train, X_test, L, torch.from_numpy(M)).numpy()
        preds = (sol @ K_test).T

        y_pred = torch.from_numpy(preds)
        preds = torch.argmax(y_pred, dim=-1)
        labels = torch.argmax(y_test, dim=-1)
        count = torch.sum(labels == preds).numpy()

        old_test_acc = count / len(labels)

        if old_test_acc > best_acc:
            best_iter = i
            best_acc = old_test_acc
            best_M = M
        M  = get_grads(X_train, sol, L, torch.from_numpy(M))

    return best_acc, best_iter, best_M


def train(X_train, y_train, X_test, y_test, c, M,
          iters=5, reg=0, L=10, normalize=False):

    y_t_orig = y_train
    y_v_orig = y_test
    y_train = convert_one_hot(y_train, c)
    y_test = convert_one_hot(y_test, c)

    if normalize:
        X_train /= norm(X_train, axis=-1).reshape(-1, 1)
        X_test /= norm(X_test, axis=-1).reshape(-1, 1)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    K_train = laplace_kernel_M(X_train, X_train, L, torch.from_numpy(M)).numpy()
    sol = solve(K_train + reg * np.eye(len(K_train)), y_train).T

    K_test = laplace_kernel_M(X_train, X_test, L, torch.from_numpy(M)).numpy()
    preds = (sol @ K_test).T

    y_pred = torch.from_numpy(preds)
    preds = torch.argmax(y_pred, dim=-1)
    labels = torch.argmax(y_test, dim=-1)
    count = torch.sum(labels == preds).numpy()

    acc = count / len(labels)
    return acc
