import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import argparse
import math
import time
import os
import numpy as np
from Modelo import*

def importar_dados():
	   # Create the model directory if does not exist
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Normalize function so given an image of range [0, 1] transforms it into a Tensor range [-1. 1]
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR LOADER
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

def model():
    modeloFC= FC_model.CoreFC(32, 100) 
	