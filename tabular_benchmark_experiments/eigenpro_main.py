import argparse
import os
import math
import numpy as np
import eigenpro_rfm as rfm


parser = argparse.ArgumentParser()
parser.add_argument('-dir', default = "data", type = str, help = "data directory")
parser.add_argument('-file', default = "result.log", type = str, help = "Output File")

args = parser.parse_args()

datadir = args.dir

avg_acc_list = []
outf = open(args.file, "w")
print ("Dataset\tSize\tNumFeatures\tNumClasses\tValidation Acc\tEpIter\tIter\tTest Acc", file = outf)

max_iter = 5
ep_iter = 50
normalize = [True, False]

for idx, dataset in enumerate(sorted(os.listdir(datadir))):
    if not os.path.isdir(datadir + "/" + dataset):
        continue
    if not os.path.isfile(datadir + "/" + dataset + "/" + dataset + ".txt"):
        continue
    dic = dict()
    for k, v in map(lambda x : x.split(), open(datadir + "/" + dataset + "/" + dataset + ".txt", "r").readlines()):
        dic[k] = v
    c = int(dic["n_clases="])
    d = int(dic["n_entradas="])
    n_train = int(dic["n_patrons_entrena="])
    n_val = int(dic["n_patrons_valida="])
    n_train_val = int(dic["n_patrons1="])
    n_test = 0
    if "n_patrons2=" in dic:
        n_test = int(dic["n_patrons2="])
    n_tot = n_train_val + n_test

    if n_tot < 100000:
        continue
    print (idx, dataset, "\tN:", n_tot, "\td:", d, "\tc:", c)

    # load data
    f = open(datadir + dataset + "/" + dic["fich1="], "r").readlines()[1:]
    X = np.asarray(list(map(lambda x: list(map(float, x.split()[1:-1])), f)))
    y = np.asarray(list(map(lambda x: int(x.split()[-1]), f)))

    # Hyperparameter Selection
    fold = list(map(lambda x: list(map(int, x.split())), open(datadir + "/" + dataset + "/" + "conxuntos.dat", "r").readlines()))
    train_fold, val_fold = fold[0], fold[1]

    best_acc, best_iter, best_M = 0, 0, None
    best_ep_iter = 0
    best_normalize = False
    print("Cross Validating")

    for n in normalize:
        acc, iter_v, M, ep_iter = rfm.hyperparam_train(X[train_fold], y[train_fold],
                                                       X[val_fold], y[val_fold], c,
                                                       ep_iter=ep_iter,
                                                       iters=max_iter, normalize=n)
        if best_M is None:
            best_M = M
        if acc > best_acc:
            best_acc = acc
            best_iter = iter_v
            best_M = M
            best_normalize = n
            best_ep_iter = ep_iter
    print("Ep iter: ", best_ep_iter)
    # 4-fold cross-validating
    avg_acc = 0.0
    fold = list(map(lambda x: list(map(int, x.split())),
                    open(datadir + dataset + "/" + "conxuntos_kfold.dat", "r").readlines()))
    print("Training")
    for repeat in range(4):
        train_fold, test_fold = fold[repeat * 2], fold[repeat * 2 + 1]

        acc = rfm.train(X[train_fold], y[train_fold], X[test_fold], y[test_fold],
                        c, best_M, iters=best_iter, ep_iter=best_ep_iter,
                        normalize=best_normalize)

        avg_acc += 0.25 * acc

    print ("acc:", avg_acc, best_ep_iter, best_iter, best_normalize,"\n")
    print(str(dataset) + '\t' + str(n_tot) + '\t' + str(d) + '\t' + str(c) + '\t' + str(best_acc) + '\t' + str(best_ep_iter) + \
          '\t' + str(best_iter) + '\t' + str(avg_acc * 100), file=outf, flush=True)

    avg_acc_list.append(avg_acc)


print ("avg_acc:", np.mean(avg_acc_list) * 100)
outf.close()
