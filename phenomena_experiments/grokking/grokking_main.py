import torch
import torchvision
import torchvision.transforms as transforms
import trainer as t
from torch.utils.data import Dataset
import random
import torch.backends.cudnn as cudnn
import rfm
import numpy as np
from sklearn.model_selection import train_test_split
from torch.linalg import norm
from random import randint
import visdom
import eigenpro_rtfm as erfm
import hickle
import neural_model

vis = visdom.Visdom('http://127.0.0.1', use_incoming_socket=False)
vis.close(env='main')

SEED = 5636
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

SIZE = 96
h, w = SIZE, SIZE
locationx = np.array([randint(0, h-1) for i in range(10)], dtype='int')
locationy = np.array([randint(0, w-1) for i in range(10)], dtype='int')

shiftx_l = np.array(locationx - 2, dtype='int')
shiftx_r = np.array(locationx + 3, dtype='int')
shifty_l = np.array(locationy - 2, dtype='int')
shifty_r = np.array(locationy + 3, dtype='int')


def one_hot_data(dataset, num_samples=-1):
    labelset = {}
    for i in range(10):
        one_hot = torch.zeros(10)
        one_hot[i] = 1
        labelset[i] = one_hot

    subset = [(ex, label) for idx, (ex, label) in enumerate(dataset) \
              if idx < num_samples and label == 0 or label == 9]

    adjusted = []

    count = 0
    for idx, (ex, label) in enumerate(subset):
        ex[:, 2:7, 7:12] = 0.
        if label == 9:
            count += 1
            ex[:, 2:7, 7:12] = 1.
        if idx < 10:
            vis.image(ex)
        ex = ex.flatten()
        adjusted.append((ex, labelset[label]))
    return adjusted

def split(trainset, p=.8):
    train, val = train_test_split(trainset, train_size=p)
    return train, val

def load_from_net(SIZE=64, path='./nn_models/trained_nn.pth'):
    dim = 3 * SIZE * SIZE
    net = neural_model.Net(dim, num_classes=10)

    d = torch.load(path)
    net.load_state_dict(d['state_dict'])
    for idx, p in enumerate(net.parameters()):
        if idx == 0:
            M = p.data.numpy()

    M = M.T @ M
    return M

def main():
    cudnn.benchmark = True
    global SIZE

    transform = transforms.Compose(
        [transforms.ToTensor()
        ])

    path = '~/datasets/'
    trainset = torchvision.datasets.STL10(root=path,
                                          split='train',
                                          transform=transform,
                                          download=True)

    trainset = one_hot_data(trainset, num_samples=500)
    trainset, valset = split(trainset, p=.8)

    print("Train Size: ", len(trainset), "Val Size: ", len(valset))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024,
                                              shuffle=True, num_workers=2)

    valloader = torch.utils.data.DataLoader(valset, batch_size=100,
                                            shuffle=False, num_workers=1)

    testset = torchvision.datasets.STL10(root=path,
                                         split='test',
                                         transform=transform,
                                         download=True)
    testset = one_hot_data(testset, num_samples=1e10)
    print(len(testset))

    testloader = torch.utils.data.DataLoader(testset, batch_size=1024,
                                             shuffle=False, num_workers=2)


    name = 'grokking'
    rfm.rfm(trainloader, valloader, testloader,
            name=name,
            iters=5,
            train_acc=True, reg=1e-3)

    t.train_network(trainloader, valloader, testloader,
                    name=name)


if __name__ == "__main__":
    main()
