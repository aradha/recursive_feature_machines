import torch
import torchvision
import torchvision.transforms as transforms
import trainer as t
from torch.utils.data import Dataset
import random
import torch.backends.cudnn as cudnn
import rfm
import numpy as np
import visdom
import numpy as np
from sklearn.model_selection import train_test_split
from torch.linalg import norm

vis = visdom.Visdom('http://127.0.0.1', use_incoming_socket=False)
vis.close(env='main')

def one_hot_data(dataset, num_samples=-1):
    labelset = {}
    for i in range(10):
        one_hot = torch.zeros(10)
        one_hot[i] = 1
        labelset[i] = one_hot

    subset = [(ex.flatten(), labelset[label]) for \
              idx, (ex, label) in enumerate(dataset) if idx < num_samples]
    return subset


def group_by_class(dataset):
    labelset = {}
    for i in range(10):
        labelset[i] = []
    for i, batch in enumerate(dataset):
        img, label = batch
        labelset[label].append(img.view(1, 3, 32, 32))
    return labelset


def merge_data(cifar, mnist, n):
    cifar_by_label = group_by_class(cifar)

    mnist_by_label = group_by_class(mnist)

    merged_data = []
    merged_labels = []

    labelset = {}

    for i in range(10):
        one_hot = torch.zeros(1, 10)
        one_hot[0, i] = 1
        labelset[i] = one_hot

    for l in cifar_by_label:

        cifar_data = torch.cat(cifar_by_label[l])
        mnist_data = torch.cat(mnist_by_label[l])
        min_len = min(len(mnist_data), len(cifar_data))
        m = min(n, min_len)
        cifar_data = cifar_data[:m]
        mnist_data = mnist_data[:m]

        merged = torch.cat([cifar_data, mnist_data], axis=-1)
        for i in range(3):
            vis.image(merged[i])
        merged_data.append(merged.reshape(m, -1))
        print(merged.shape)
        merged_labels.append(np.repeat(labelset[l], m, axis=0))
    merged_data = torch.cat(merged_data, axis=0)

    merged_labels = np.concatenate(merged_labels, axis=0)
    merged_labels = torch.from_numpy(merged_labels)

    return list(zip(merged_data, merged_labels))

def split(trainset, p=.8):
    train, val = train_test_split(trainset, train_size=p)
    return train, val

def main():
    SEED = 5636
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    cudnn.benchmark = True

    transform = transforms.Compose(
        [transforms.ToTensor()
        ])

    mnist_transform = transforms.Compose(
        [transforms.Resize([32,32]),
         transforms.ToTensor(),
         transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])


    path = '~/datasets/'
    cifar_trainset = torchvision.datasets.CIFAR10(root=path,
                                                  train=True,
                                                  transform=transform,
                                                  download=False)

    mnist_trainset = torchvision.datasets.MNIST(root=path,
                                                train=True,
                                                transform=mnist_transform,
                                                download=False)

    trainset = merge_data(cifar_trainset, mnist_trainset, 5000)
    trainset, valset = split(trainset, p=.8)
    print("Train Size: ", len(trainset), "Val Size: ", len(valset))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=100,
                                            shuffle=False, num_workers=1)

    cifar_testset = torchvision.datasets.CIFAR10(root=path,
                                                 train=False,
                                                 transform=transform,
                                                 download=False)

    mnist_testset = torchvision.datasets.MNIST(root=path,
                                               train=False,
                                               transform=mnist_transform,
                                               download=False)

    testset = merge_data(cifar_testset, mnist_testset, 1000)
    print("Test Size: ", len(testset))

    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=2)

    name = 'cifar_mnist'
    rfm.rfm(trainloader, valloader, testloader, name=name,
            batch_size=10, iters=1, reg=1e-3)

    t.train_network(trainloader, valloader, testloader,
                    num_classes=10,
                    name=name)


if  __name__ == "__main__":
    main()
