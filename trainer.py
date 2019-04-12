import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from data_load import FacialKeypointsDataset
from data_load import Rescale, RandomCrop, Normalize, ToTensor

from models import Net_Base

def train_net(neural_net, n_epochs, criterion, train_loader, optimizer_type, lr):
    if optimizer_type == "Adam":
        print("Adam, lr = ", lr)
        optimizer = optim.Adam(params=neural_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    elif optimizer_type == "SGD":
        print("SGD, lr = ", lr)
        optimizer = optim.SGD(params=neural_net.parameters(), lr=lr)
    elif optimizer_type == "RMS":
        print("RMS, lr = ", lr)
        optimizer = optim.RMSprop(params=neural_net.parameters(), lr=0.01)
    else:
        raise("unknown optimizer")

    # prepare the net for training
    neural_net.train()

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            # forward pass to get outputs
            output_pts = neural_net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            # to convert loss into a scalar and add it to the running_loss, use .item()
            running_loss += loss.item()
            if batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/1000))
                running_loss = 0.0

    print('Finished Training')

data_transform = transforms.Compose([Rescale(250), RandomCrop(224), Normalize(), ToTensor()])
transformed_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv', root_dir='data/training/', transform=data_transform)

for batch in [10,20,30]:
    print ("Batch:", batch)
    train_dataloader = DataLoader(transformed_dataset, batch_size=batch, shuffle=True, num_workers=4)

    train_net(Net_Base(), 1, nn.MSELoss(), train_dataloader, "Adam", 0.001)
    train_net(Net_Base(), 1, nn.MSELoss(), train_dataloader, "Adam", 0.005)
    train_net(Net_Base(), 1, nn.MSELoss(), train_dataloader, "Adam", 0.01)
    train_net(Net_Base(), 1, nn.MSELoss(), train_dataloader, "SGD", 0.001)
    train_net(Net_Base(), 1, nn.MSELoss(), train_dataloader, "SGD", 0.005)
    train_net(Net_Base(), 1, nn.MSELoss(), train_dataloader, "SGD", 0.01)
    train_net(Net_Base(), 1, nn.MSELoss(), train_dataloader, "RMS", 0.001)
    train_net(Net_Base(), 1, nn.MSELoss(), train_dataloader, "RMS", 0.005)
    train_net(Net_Base(), 1, nn.MSELoss(), train_dataloader, "RMS", 0.01)
