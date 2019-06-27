#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import argparse
import logging

from utils import *
from cnn import Net


logger = logging.getLogger("sample_pytorch_mnist")
parser = argparse.ArgumentParser(description='Command line entry to pytorch example')

parser.add_argument(
    "--load", default=False, help="Load model", type=bool)
parser.add_argument(
    "--train",
    default=False,
    help="Train then save model)",
    type=bool)

parser.add_argument(
    "--log_level", default="info", help="The log level(eg. info)")

args = parser.parse_args()

if args.log_level == "info" or args.log_level == "INFO":
  logger.setLevel(logging.INFO)
elif args.log_level == "debug" or args.log_level == "DEBUG":
  logger.setLevel(logging.DEBUG)
elif args.log_level == "error" or args.log_level == "ERROR":
  logger.setLevel(logging.ERROR)
elif args.log_level == "warning" or args.log_level == "WARNING":
  logger.setLevel(logging.WARNING)
elif args.log_level == "critical" or args.log_level == "CRITICAL":
  logger.setLevel(logging.CRITICAL)

for arg in vars(args):
    logger.info("{}: {}".format(arg, getattr(args, arg)))


##
torch.backends.cudnn.enabled = False
momentum = 0.5
learning_rate = 0.01
n_epochs = 3

training_data, testing_data, example_data, example_targets  = getData()

# Save
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
loss = 0

test(network, testing_data)
for epoch in range(1, n_epochs + 1):
    train(epoch, network, optimizer, loss, training_data)
    test(network, testing_data)
    

# Load
continued_network = Net()
continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                                momentum=momentum)
continued_loss = 0
load(continued_network, continued_optimizer, "model.pth", training=True, useGPU=False)

for i in range(4,9):
    train(i, continued_network, continued_optimizer, continued_loss, training_data)
    test(continued_network, testing_data)


