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


# https://necromuralist.github.io/In-Too-Deep/posts/nano/pytorch/part-6-saving-and-loading-models/#org5b57813

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')






# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))



net = Net()
criterion = nn.CrossEntropyLoss()
optimizerpt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizerpt.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizerpt.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            try: 
                print ("Checkpoint - Saving model at {}".format(epoch))
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizerpt.state_dict(),
                        'loss': loss,
                        
                        }, "model.pth")
            except:
                print("failed to save")


print('Finished Training')


print (net.state_dict())
# torch.save_state_dict(net)

# checkpoint = {'model': Net(),
#           'state_dict': model.state_dict(),
#           'optimizer' : optimizer.state_dict()}

# torch.save(checkpoint, 'model.pth')

print('Finished Saving')


