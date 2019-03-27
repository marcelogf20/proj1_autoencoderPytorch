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
from Modelos import*

def main(model,coded_size, patch_size,num_epochs,batch_size,lr_rate,w_d,log_step,model_path,save_step):
	   # Create the model directory if does not exist
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Normalize function so given an image of range [0, 1] transforms it into a Tensor range [-1. 1]
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR LOADER
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    modeloFC= FC_modelo.CoreFC(coded_size, patch_size)
    print(modeloFC)
   
     # Define the LOSS and the OPTIMIZER
    criterion = nn.MSELoss()
    params = list(modeloFC.parameters())
    print(len(params))
    optimizer = optim.Adam(params, lr=lr_rate, weight_decay=w_d)

    # ::::::::::::::::::::::::::::::::
    #   TRAIN----------------------
    # ::::::::::::::::::::::::::::::::

    num_steps = len(train_loader)
    start = time.time()
    total_losses = []
    # Divide the input 32x32 images into num_patches patch_sizexpatch_size patchs
    num_patches = (32//patch_size)**2
    print((num_patches))

    for epoch in range(num_epochs):

        running_loss = 0.0
        current_losses = []
        for i, data in enumerate(train_loader, 0):

            # Get the images
            imgs = data[0]

            # Transform into patches
            patches = to_patches(imgs, patch_size)
            if i==0:
                print(patches[0])
            # TODO: Do this thing more polite!! :S
            for patch in patches:
                # Transform the tensor into Variable
                v_patch = Variable(patch)
                target_tensor = Variable(torch.zeros(v_patch.size()), requires_grad=False)
                losses = []
                    # Set gradients to Zero
                optimizer.zero_grad()
                reconstructed_patches = modeloFC(v_patch)
                loss = criterion(reconstructed_patches, v_patch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            

            # STATISTICS:

            if (i+1) % log_step == 0:
                print('(%s) [%d, %5d] loss: %.3f' %
                      (timeSince(start, ((epoch * num_steps + i + 1.0) / (num_epochs * num_steps))),
                       epoch + 1, i + 1, running_loss / log_step / num_patches))
                current_losses.append(running_loss/log_step/num_patches)
                running_loss = 0.0

            # SAVE:
            if (i + 1) % save_step == 0:
                torch.save(modeloFC.state_dict(),
                           os.path.join(model_path, model+'-p%d_b%d-%d_%d.pkl' %
                                        (patch_size, coded_size, epoch + 1, i + 1)))

        total_losses.append(current_losses)
        torch.save(modeloFC.state_dict(),
                   os.path.join(model_path,
                                model + '-p%d_b%d-%d_%d.pkl' % (patch_size, coded_size, epoch + 1, i + 1)))

    print('__TRAINING DONE=================================================')


#==============================================
# - CUSTOM FUNCTIONS
#==============================================

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def to_patches(x, patch_size):
    num_patches_x = 32//patch_size
    patches = []
    for i in range(num_patches_x):
        for j in range(num_patches_x):
            patch = x[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patches.append(patch.contiguous())
    return patches


if __name__ == '__main__':
    main('fc',4, 8,1,4,0.01,0,10,'./saved_models/',50)
	