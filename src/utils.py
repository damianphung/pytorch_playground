from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision


def train(epoch, model, optimizer, loss, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        log_interval = 10
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # train_losses.append(loss.item())
            # train_counter.append(
            #     (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            save(epoch, model, optimizer, loss, "model.pth")
            # torch.save(model.state_dict(), 'model.pth')
            # torch.save(optimizer.state_dict(), 'optimizer.pth')


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    # test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))




def getData():
    random_seed = 1
    batch_size_train = 64
    batch_size_test = 1000    
    torch.manual_seed(random_seed)
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('data/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('data/', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=batch_size_test, shuffle=True)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    return ( train_loader, test_loader, example_data, example_targets )

def save(epoch, model, optimizer, loss, filepath):
    state = {
      'epoch': epoch,
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'loss': loss
    }
    torch.save(state, filepath)

def load(model, optimizer, filepath, training=True, useGPU=False):
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    # if useGPU == False:
    #     device = torch.device("cpu")
    # else:
    #     device = torch.device("cuda")

    state = torch.load(filepath)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    epoch = state['epoch']
    loss = state['loss']

    if training == True:
        model.train()
    else:
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()