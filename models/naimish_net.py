"""
Implementation of the NaimishNet

more information: https://arxiv.org/pdf/1710.00977.pdf
"""

import torch.nn as nn
import torch.nn.functional as F


class NaimishNet(nn.Module):
    def __init__(self):
        super(NaimishNet, self).__init__()

        """
        from the paper:
        layer | number of filters | filter shape|
        1     |      32           |     (4,4)   
        2     |      64           |     (3,3)   
        3     |      128          |     (2,2)     
        4     |      256          |     (1,1)   

        Activation : ELU
        Dropout: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
        MaxPool = 4x(2,2)
        Flatten: flattens 3d input to 1d output
        Dense1 to Dense3 are regular fully connected layers with weights initialized using Glorot uniform initialization
        Adam optimizer, with lr = 0.001, Beta1 of 0.9, Beta2 of 0.999 and Epsilon of 1e−08, is used for minimizing MSE
        """
        # Image resolution 224x224
        # For conv layer out :     (W-K)/S + 1
        # For MaxPool layer out :  W/2

        self.max_pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 32, 4)  # (224-4)/1 + 1 = 221 -> (32, 221, 221)
        # after max pool => (32, 110, 110)
        self.drop1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv2d(32, 64, 3)  # (110-3)/1 + 1 = 108 -> (64, 108, 108)
        # after max pool => (64, 54, 54)
        self.drop2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(64, 128, 2)  # (54-2)/1 + 1 = 53 -> (128, 53, 53)
        # after max pool => (128, 26, 26)
        self.drop3 = nn.Dropout(0.3)

        self.conv4 = nn.Conv2d(128, 256, 1)  # (26-1)/1 + 1 = 26 -> (256, 26, 26)
        # after max pool => (256, 13, 13)
        self.drop4 = nn.Dropout(0.4)

        self.dense1 = nn.Linear(256 * 13 * 13, 256 * 13)
        self.drop5 = nn.Dropout(0.5)

        self.dense2 = nn.Linear(256 * 13, 256 * 13)
        self.drop6 = nn.Dropout(0.6)

        self.dense3 = nn.Linear(256 * 13, 136)

    def forward(self, x):
        x = self.max_pool(F.elu(self.conv1(x)))
        x = self.drop1(x)

        x = self.max_pool(F.elu(self.conv2(x)))
        x = self.drop2(x)

        x = self.max_pool(F.elu(self.conv3(x)))
        x = self.drop3(x)

        x = self.max_pool(F.elu(self.conv4(x)))
        x = self.drop4(x)

        # flatten layer
        x = x.view(x.size(0), -1)

        x = F.elu(self.dense1(x))
        x = self.drop5(x)

        x = F.relu(self.dense2(x))
        x = self.drop6(x)

        x = self.dense3(x)

        return x
