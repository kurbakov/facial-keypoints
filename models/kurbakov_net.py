"""
The customized model, based on the NaimishNet
"""

import torch.nn as nn
import torch.nn.functional as F


class KurbakovNet(nn.Module):
    def __init__(self):
        super(KurbakovNet, self).__init__()
        self.max_pool = nn.MaxPool2d(2,2)
        self.conv1 = nn.Conv2d(1, 32, 4)    # (224-4)/1 + 1 = 221 -> (32, 221, 221)
        # after max pool => (32, 110, 110)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 4)   # (110-4)/1 + 1 = 107 -> (64, 107, 107)
        # after max pool => (64, 53, 53)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 4)  # (53-4)/1 + 1 = 50 -> (128, 50, 50)
        # after max pool => (128, 25, 25)
        self.bn3 = nn.BatchNorm2d(128)
                
        self.dense1 = nn.Linear(128*25*25, 256*13)
        self.drop5 = nn.Dropout(0.2)
        
        self.dense2 = nn.Linear(256*13, 256*13)
        self.drop6 = nn.Dropout(0.2)
        
        self.dense3 = nn.Linear(256*13, 136)        

    def forward(self, x):
        x = self.max_pool(F.elu(self.bn1(self.conv1(x))))
        x = self.max_pool(F.elu(self.bn2(self.conv2(x))))
        x = self.max_pool(F.elu(self.bn3(self.conv3(x))))

        # flatten layer
        x = x.view(x.size(0), -1)
        
        x = F.elu(self.dense1(x))
        x = self.drop5(x)

        x = F.relu(self.dense2(x))
        x = self.drop6(x)
        
        x = self.dense3(x)
        
        return x

