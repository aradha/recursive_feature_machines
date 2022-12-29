import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
import torch.backends.cudnn as cudnn
import rfm
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import hickle
from torch.linalg import norm
import visdom

vis = visdom.Visdom('http://127.0.0.1', use_incoming_socket=False)
vis.close(env='avg_vis')

NUM_CLASSES = 2

def get_data(dataset, mask):

    labelset = {}
    for i in range(NUM_CLASSES):
        one_hot = torch.zeros(NUM_CLASSES)
        one_hot[i] = 1
        labelset[i] = one_hot
    # Full list of indices and labels
    """
    0 5_o_Clock_Shadow
    1 Arched_Eyebrows
    2 Attractive
    3 Bags_Under_Eyes
    4 Bald
    5 Bangs
    6 Big_Lips
    7 Big_Nose
    8 Black_Hair
    9 Blond_Hair
    10 Blurry
    11 Brown_Hair
    12 Bushy_Eyebrows
    13 Chubby
    14 Double_Chin
    15 Eyeglasses
    16 Goatee
    17 Gray_Hair
    18 Heavy_Makeup
    19 High_Cheekbones
    20 Male
    21 Mouth_Slightly_Open
    22 Mustache
    23 Narrow_Eyes
    24 No_Beard
    25 Oval_Face
    26 Pale_Skin
    27 Pointy_Nose
    28 Receding_Hairline
    29 Rosy_Cheeks
    30 Sideburns
    31 Smiling
    32 Straight_Hair
    33 Wavy_Hair
    34 Wearing_Earrings
    35 Wearing_Hat
    36 Wearing_Lipstick
    37 Wearing_Necklace
    38 Wearing_Necktie
    39 Young
    """

    feature_idx = 15
    by_class = {}
    features = []

    if num_samples is None:
        num_samples = len(dataset)
    for idx in tqdm(range(len(dataset))):
        ex, label = dataset[idx]
        features.append(label[feature_idx])
        g = label[feature_idx].numpy().item()
        ex = ex[mask]
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

    data.extend(by_class[1][:max_len])
    data.extend(by_class[0][:max_len])
    return data


def get_mask(M, t=.05):
    d, _ = M.shape
    SIZE = int(np.sqrt(d // 3))
    F1 = np.diag(M[:SIZE**2, :SIZE**2]).reshape(SIZE, SIZE)
    F2 = np.diag(M[SIZE**2:2*SIZE**2, SIZE**2:2*SIZE**2]).reshape(SIZE, SIZE)
    F3 = np.diag(M[2*SIZE**2:, 2*SIZE**2:]).reshape(SIZE, SIZE)
    F = np.stack([F1, F2, F3])
    print(F.shape)

    F = (F - F.min()) / (F.max() - F.min())

    vis.image(F, env='avg_vis')
    for i in range(len(F)):
        F[i] = np.where(F[i] > t, 1, 0)

    coords = F.nonzero()
    vis.image(F, env='avg_vis')
    return coords


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

    # Select an M matrix
    M = hickle.load('path_to_M_matrix')
    mask = get_mask(M)
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


    trainset = get_data(trainset, mask)

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

    testset = get_data(testset, mask)
    print("Test Size: ", len(testset))

    testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                             shuffle=False, num_workers=1)

    name = 'lottery_ticket'
    # Re-train using laplace kernel (iteration 0 of RFM)
    rfm.rfm(trainloader, valloader, testloader, name=name, reg=1e-3,
            iters=0)


if  __name__ == "__main__":
    main()
