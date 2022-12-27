import numpy as np
import torch
from numpy.linalg import solve
import classic_kernel
from tqdm import tqdm
import hickle


def laplace_kernel_M(pair1, pair2, bandwidth, M):
    return classic_kernel.laplacian_M(pair1, pair2, bandwidth, M)


def get_grads(X, sol, L, P, batch_size=2):
    M = 0.

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

    bs = batch_size
    batches = torch.split(G, bs)
    for i in tqdm(range(len(batches))):
        grad = batches[i].cuda()
        gradT = torch.transpose(grad, 1, 2)
        M += torch.sum(gradT @ grad, dim=0).cpu()
        del grad, gradT
    torch.cuda.empty_cache()
    M /= len(G)
    M = M.numpy()

    return M


def rfm(train_loader, val_loader, test_loader,
        iters=3, name=None, batch_size=2, reg=1e-3,
        train_acc=False):

    L = 10

    X_train, y_train = get_data(train_loader)
    X_val, y_val = get_data(val_loader)
    X_test, y_test = get_data(test_loader)

    n, d = X_train.shape

    M = np.eye(d, dtype='float32')

    for i in range(iters):
        K_train = laplace_kernel_M(X_train, X_train, L, torch.from_numpy(M)).numpy()
        sol = solve(K_train + reg * np.eye(len(K_train)), y_train).T

        if train_acc:
            preds = (sol @ K_train).T
            y_pred = torch.from_numpy(preds)
            preds = torch.argmax(y_pred, dim=-1)
            labels = torch.argmax(y_train, dim=-1)
            count = torch.sum(labels == preds).numpy()
            print("Round " + str(i) + " Train Acc: ", count / len(labels))

        K_test = laplace_kernel_M(X_train, X_test, L, torch.from_numpy(M)).numpy()
        preds = (sol @ K_test).T
        print("Round " + str(i) + " MSE: ", np.mean(np.square(preds - y_test.numpy())))
        y_pred = torch.from_numpy(preds)
        preds = torch.argmax(y_pred, dim=-1)
        labels = torch.argmax(y_test, dim=-1)
        count = torch.sum(labels == preds).numpy()
        print("Round " + str(i) + " Acc: ", count / len(labels))

        M  = get_grads(X_train, sol, L, torch.from_numpy(M), batch_size=batch_size)
        if name is not None:
            hickle.dump(M, 'saved_Ms/M_' + name + '_' + str(i) + '.h')

    K_train = laplace_kernel_M(X_train, X_train, L, torch.from_numpy(M)).numpy()
    sol = solve(K_train + reg * np.eye(len(K_train)), y_train).T
    K_test = laplace_kernel_M(X_train, X_test, L, torch.from_numpy(M)).numpy()
    preds = (sol @ K_test).T
    mse = np.mean(np.square(preds - y_test.numpy()))
    print("Final MSE: ", mse)
    y_pred = torch.from_numpy(preds)
    preds = torch.argmax(y_pred, dim=-1)
    labels = torch.argmax(y_test, dim=-1)
    count = torch.sum(labels == preds).numpy()
    print(" Final Acc: ", count / len(labels))
    return mse


def get_data(loader):
    X = []
    y = []
    for idx, batch in enumerate(loader):
        inputs, labels = batch
        X.append(inputs)
        y.append(labels)
    return torch.cat(X, dim=0), torch.cat(y, dim=0)
