#!/usr/bin/env python
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from cnn import Net
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from collections import OrderedDict

net = Net()
optimizerpt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)  
    print ("loading...") 
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizerpt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print ("checkpoint loaded")

    for parameter in net.parameters():
        print ( parameter )
        parameter.requires_grad = False

    net.eval()
    return net


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
try:
    load_checkpoint("model.pth")
    # torch.load_state_dict(torch.load("model.pth"))
    # net = load_checkpoint("model.pth")
except:
    print ("Failed to load" )
    exit (0)


dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
# imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
outputs = net(images)


_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))