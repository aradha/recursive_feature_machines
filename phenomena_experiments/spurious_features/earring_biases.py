import hickle
import torch
import numpy as np
import neural_model
import random
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import visdom
from torch.linalg import norm
import csv

vis = visdom.Visdom('http://127.0.0.1', use_incoming_socket=False)
vis.close(env='avg_vis')
vis.close(env='main')
def get_data(dataset, mask=None, num_samples=-1, balanced=False,
             break_idx=1):

    NUM_CLASSES = 2
    # Make balanced classes
    labelset = {}
    for i in range(NUM_CLASSES):
        one_hot = torch.zeros(NUM_CLASSES)
        one_hot[i] = 1
        labelset[i] = one_hot

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

    feature_idx = 34
    male_idx = 20
    by_class = {}
    features = []
    count = 0
    male_count = 0.
    template = None

    break_count = 0
    for idx in tqdm(range(len(dataset))):
        ex, label = dataset[idx]
        features.append(label[feature_idx])
        g = label[feature_idx].numpy().item()
        m = label[male_idx].numpy().item()
        if m == 1 and g != 1:
            # Uncomment to use RFM mask
            #template = ex * mask

            # Uncomment to use ear mask
            #template = ex
            #template[:, :, :30] = ex[:, :, :30]
            #template[:, :, 60:] = ex[:, :, 60:]
            #template[:, :, 30:60] = 0.

            break_count += 1
            if break_count == break_idx:
                break

    for idx in tqdm(range(len(dataset))):
        ex, label = dataset[idx]
        features.append(label[feature_idx])
        g = label[feature_idx].numpy().item()
        m = label[male_idx].numpy().item()

        # Uncomment for RFM mask
        #ex = np.where(mask, template, ex)
        # Uncomment for ear mask
        #ex[:, :, :30] = template[:, :, :30]
        #ex[:, :, 60:] = template[:, :, 60:]

        if m == 1:
            male_count += 1
        count += 1
        if count <= 20:
            tex = (ex - ex.min()) / (ex.max() - ex.min())
            vis.image(tex)
        if g in by_class:
            by_class[g].append((ex.flatten(), labelset[g]))
        else:
            by_class[g] = [(ex.flatten(), labelset[g])]
        if idx > num_samples:
            break

    data = []
    data.extend(by_class[1])
    data.extend(by_class[0][:len(by_class[1])])
    return data

def get_acc(net, loader):
    net.eval()
    count = 0
    for batch_idx, batch in enumerate(loader):
        inputs, targets = batch
        with torch.no_grad():
            output = net(Variable(inputs).cuda())
            target = Variable(targets).cuda()#.view(-1, 1)
        preds = torch.argmax(output, dim=-1)
        labels = torch.argmax(target, dim=-1)
        count += torch.sum(labels == preds).cpu().data.numpy()
    return count / len(loader.dataset) * 100


def visualize_M(M, t1=.05, t2=1):
    d, _ = M.shape
    SIZE = int(np.sqrt(d // 3))
    F1 = np.diag(M[:SIZE**2, :SIZE**2]).reshape(SIZE, SIZE)
    F2 = np.diag(M[SIZE**2:2*SIZE**2, SIZE**2:2*SIZE**2]).reshape(SIZE, SIZE)
    F3 = np.diag(M[2*SIZE**2:, 2*SIZE**2:]).reshape(SIZE, SIZE)
    F = np.stack([F1, F2, F3])
    print(F.shape)
    t = np.quantile(F.reshape(-1), .99)
    F = (F - F.min()) / (F.max() - F.min())
    vis.image(F, env='avg_vis')
    for i in range(len(F)):
        F[i] = np.where((F[i] > t1) & (F[i] < t2), 1, 0)
    F = np.max(F, axis=0)
    vis.image(F, env='avg_vis')
    return F


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


    dim = 3 * SIZE * SIZE
    net = neural_model.Net(dim)
    path = 'path to earring nn model'

    d = torch.load(path)
    net.load_state_dict(d['state_dict'])

    for idx, p in enumerate(net.parameters()):
        if idx == 0:
            M = p.data.numpy()

    M = M.T @ M
    M_path = 'path to RFM earring model M'

    M = hickle.load(M_path)
    coords = visualize_M(M)

    mask = coords != 0

    # Change to see effect of other masks
    num_samples = 1
    break_indices = list(range(num_samples))

    celeba_path = '~/datasets/'

    testset = torchvision.datasets.CelebA(root=celeba_path,
                                          split='test',
                                          transform=transform,
                                          download=False)

    # Writing to log file
    f = open('csv_logs/earring_log.csv', 'w')
    writer = csv.writer(f, delimiter=',')
    for break_idx in break_indices:
        test = get_data(testset, mask=mask, num_samples=1e10,
                        break_idx=break_idx+1)


        testloader = torch.utils.data.DataLoader(test, batch_size=512,
                                                 shuffle=False, num_workers=1)

        net.eval()
        net.cuda()
        test_acc = get_acc(net, testloader)
        print("Test acc: ", test_acc)
        row = [test_acc]
        writer.writerow(row)
        f.flush()

if __name__ == "__main__":
    main()
