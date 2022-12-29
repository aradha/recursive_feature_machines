import torch
from torch.autograd import Variable
import torch.optim as optim
import time
import neural_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def visualize_M(M, idx):
    d, _ = M.shape
    SIZE = int(np.sqrt(d // 3))
    F1 = np.diag(M[:SIZE**2, :SIZE**2]).reshape(SIZE, SIZE)
    F2 = np.diag(M[SIZE**2:2*SIZE**2, SIZE**2:2*SIZE**2]).reshape(SIZE, SIZE)
    F3 = np.diag(M[2*SIZE**2:, 2*SIZE**2:]).reshape(SIZE, SIZE)
    F = np.stack([F1, F2, F3])
    print(F.shape)
    F = (F - F.min()) / (F.max() - F.min())
    F = np.rollaxis(F, 0, 3)
    plt.imshow(F)
    plt.axis('off')
    plt.savefig('./video_logs/' + str(idx).zfill(6) + '.png',
                bbox_inches='tight', pad_inches = 0)
    return F


def train_network(train_loader, val_loader, test_loader,
                  num_classes=2, name=None,
                  save_frames=False):


    for idx, batch in enumerate(train_loader):
        inputs, labels = batch
        _, dim = inputs.shape
        break
    net = neural_model.Net(dim, num_classes=num_classes)

    params = 0
    for idx, param in enumerate(list(net.parameters())):
        size = 1
        for idx in range(len(param.size())):
            size *= param.size()[idx]
            params += size
    print("NUMBER OF PARAMS: ", params)

    optimizer = torch.optim.SGD(net.parameters(), lr=.1)

    net.cuda()
    num_epochs = 501
    best_val_acc = 0
    best_test_acc = 0
    best_val_loss = np.float("inf")
    best_test_loss = 0

    for i in range(num_epochs):
        if save_frames:
            net.cpu()
            for idx, p in enumerate(net.parameters()):
                if idx == 0:
                    M = p.data.numpy()
            M = M.T @ M
            visualize_M(M, i)
            net.cuda()

        if i == 0 or i == 1:
            net.cpu()
            d = {}
            d['state_dict'] = net.state_dict()
            if name is not None:
                torch.save(d, 'nn_models/' + name + '_trained_nn_' + str(i) + '.pth')
            else:
                torch.save(d, 'nn_models/trained_nn.pth')
            net.cuda()

        train_loss = train_step(net, optimizer, train_loader, save_frames=save_frames)
        val_loss = val_step(net, val_loader)
        test_loss = val_step(net, test_loader)
        train_acc = get_acc(net, train_loader)
        val_acc = get_acc(net, val_loader)
        test_acc = get_acc(net, test_loader)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            net.cpu()
            d = {}
            d['state_dict'] = net.state_dict()
            if name is not None:
                torch.save(d, 'nn_models/' + name + '_trained_nn.pth')
            else:
                torch.save(d, 'nn_models/trained_nn.pth')
            net.cuda()

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_test_loss = test_loss

        print("Epoch: ", i,
              "Train Loss: ", train_loss, "Test Loss: ", test_loss,
              "Train Acc: ", train_acc, "Test Acc: ", test_acc,
              "Best Val Acc: ", best_val_acc, "Best Val Loss: ", best_val_loss,
              "Best Test Acc: ", best_test_acc, "Best Test Loss: ", best_test_loss)


def get_data(loader):
    X = []
    y = []
    for idx, batch in enumerate(loader):
        inputs, labels = batch
        X.append(inputs)
        y.append(labels)
    return torch.cat(X, dim=0), torch.cat(y, dim=0)


def train_step(net, optimizer, train_loader, save_frames=False):
    net.train()
    start = time.time()
    train_loss = 0.

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, labels = batch
        targets = labels
        output = net(Variable(inputs).cuda())
        target = Variable(targets).cuda()
        loss = torch.mean(torch.pow(output - target, 2))
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().data.numpy() * len(inputs)
    end = time.time()
    print("Time: ", end - start)
    train_loss = train_loss / len(train_loader.dataset)
    return train_loss


def val_step(net, val_loader):
    net.eval()
    val_loss = 0.

    for batch_idx, batch in enumerate(val_loader):
        inputs, labels = batch
        targets = labels
        with torch.no_grad():
            output = net(Variable(inputs).cuda())
            target = Variable(targets).cuda()
        loss = torch.mean(torch.pow(output - target, 2))
        val_loss += loss.cpu().data.numpy() * len(inputs)
    val_loss = val_loss / len(val_loader.dataset)
    return val_loss


def get_acc(net, loader):
    net.eval()
    count = 0
    for batch_idx, batch in enumerate(loader):
        inputs, targets = batch
        with torch.no_grad():
            output = net(Variable(inputs).cuda())
            target = Variable(targets).cuda()

        preds = torch.argmax(output, dim=-1)
        labels = torch.argmax(target, dim=-1)

        count += torch.sum(labels == preds).cpu().data.numpy()
    return count / len(loader.dataset) * 100
