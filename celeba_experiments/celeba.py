import torch
import torchvision
import torchvision.transforms as transforms
import trainer as t
from torch.utils.data import Dataset
import random
import torch.backends.cudnn as cudnn
import rfm
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import hickle
from torch.linalg import norm

NUM_CLASSES = 2

def get_balanced_data(dataset, num_samples=None):

    if num_samples is None:
        num_samples = len(dataset)

    # Make balanced classes
    labelset = {}
    for i in range(NUM_CLASSES):
        one_hot = torch.zeros(NUM_CLASSES)
        one_hot[i] = 1
        labelset[i] = one_hot

    # All attributes found in list_attr_celeba.txt
    feature_idx = 15  # Index of feature label - 15 corresponds to glasses
    by_class = {}
    features = []
    for idx in tqdm(range(len(dataset))):
        ex, label = dataset[idx]
        features.append(label[feature_idx])
        g = label[feature_idx].numpy().item()
        ex = ex.flatten()
        ex = ex / norm(ex)
        if g in by_class:
            by_class[g].append((ex, labelset[g]))
        else:
            by_class[g] = [(ex, labelset[g])]
        if idx > num_samples:
            break
    data = []
    max_len = min(25000, len(by_class[1]))

    print(by_class.keys())
    data.extend(by_class[1][:max_len])
    data.extend(by_class[0][:max_len])
    return data


def split(trainset, p=.8):
    train, val = train_test_split(trainset, train_size=p)
    return train, val


def main():
    SEED = 5636
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    cudnn.benchmark = True

    SIZE = 96
    transform = transforms.Compose(
        [transforms.Resize([SIZE,SIZE]),
         transforms.ToTensor()
        ])

    celeba_path = '~/datasets/'
    trainset = torchvision.datasets.CelebA(root=celeba_path,
                                           split='train',
                                           transform=transform,
                                           download=True)

    trainset = get_balanced_data(trainset)
    trainset, valset = split(trainset, p=.8)

    print("Train Size: ", len(trainset), "Val Size: ", len(valset))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=1)

    valloader = torch.utils.data.DataLoader(valset, batch_size=100,
                                            shuffle=False, num_workers=1)

    testset = torchvision.datasets.CelebA(root=celeba_path,
                                          split='test',
                                          transform=transform,
                                          download=True)

    testset = get_balanced_data(testset)
    print("Test Size: ", len(testset))

    testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                             shuffle=False, num_workers=1)

    # Optional name for saving model
    name = 'glasses'

    # Code for training rfm
    rfm.rfm(trainloader, valloader, testloader, name=name, reg=1e-3,
            iters=1)

    # Code for training neural network
    t.train_network(trainloader, valloader, testloader, name=name)


if  __name__ == "__main__":
    main()
