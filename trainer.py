import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from utils.data_set import FacialKeypointsDataset
from utils.rescale import Rescale
from utils.random_crop import RandomCrop
from utils.normalize import Normalize
from utils.to_tensor import ToTensor

from models.kurbakov_net import KurbakovNet
from models.naimish_net import NaimishNet


def train_net(neural_net, n_epochs, criterion, train_loader, optimizer_type, lr):
    if optimizer_type == "Adam":
        opt = optim.Adam(params=neural_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    elif optimizer_type == "SGD":
        opt = optim.SGD(params=neural_net.parameters(), lr=lr)
    elif optimizer_type == "RMS":
        opt = optim.RMSprop(params=neural_net.parameters(), lr=0.01)
    else:
        raise Exception("unknown optimizer")

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
            opt.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            opt.step()

            # print loss statistics
            # to convert loss into a scalar and add it to the running_loss, use .item()
            running_loss += loss.item()
            if batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/1000))
                running_loss = 0.0

    print('Finished Training')


def test_multiple_setups(n_epochs, batches, optimizer_types, learning_rates):
    data_transform = transforms.Compose([Rescale(250), RandomCrop(224), Normalize(), ToTensor()])
    transformed_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                                 root_dir='data/training/',
                                                 transform=data_transform)

    # We use a for loop just to test and see how the variables influence the model accurac
    for batch in batches:
        data_loader = DataLoader(transformed_dataset, batch_size=batch, shuffle=True, num_workers=4)
        for optimizer in optimizer_types:
            for lr in learning_rates:
                print("Batch:", batch)
                print("Optimizer:", optimizer)
                print("lr:", lr)
                print("NNet:", "KurbakovNet")
                train_net(KurbakovNet(), n_epochs, nn.MSELoss(), data_loader, optimizer, lr)
                print("NNet:", "NaimishNet")
                train_net(NaimishNet(), n_epochs, nn.MSELoss(), data_loader, optimizer, lr)


def train_and_save(n_epoch, batch_size, optimizer, learning_rate, path):
    data_transform = transforms.Compose([Rescale(250), RandomCrop(224), Normalize(), ToTensor()])
    transformed_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                                 root_dir='data/training/',
                                                 transform=data_transform)

    data_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    net = KurbakovNet()
    train_net(net, n_epoch, nn.MSELoss(), data_loader, optimizer, learning_rate)

    torch.save(net.state_dict(), path)


if __name__ == '__main__':
    # try different combinations
    test_multiple_setups(1, [8, 16, 32, 64], ["Adam", "SGD", "RMS"], [0.001, 0.005, 0.01])

    # Once you know which model is the best, train it and save
    neural_net = KurbakovNet()
    train_and_save(1, 64, optim.SGD(params=neural_net.parameters(), lr=0.1), "./saved_models/my_model.pt")