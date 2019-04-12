## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

####################################    
#         Best network   
####################################
class Net_Modified(nn.Module):
    def __init__(self):
        super(Net_Modified, self).__init__()
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
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
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

class Net_Base(nn.Module):

    def __init__(self):
        super(Net_Base, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers 
        # (such as dropout or batch normalization) to avoid overfitting
        
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
        Adam optimizer, with learning rate of 0.001, Beta1 of 0.9, Beta2 of 0.999 and Epsilon of 1e−08, is used for minimizing MSE
        """
        # Image resolution 224x224
        # For conv layer out :     (W-K)/S + 1
        # For MaxPool layer out :  W/2
        
        self.max_pool = nn.MaxPool2d(2,2)

        self.conv1 = nn.Conv2d(1, 32, 4)    # (224-4)/1 + 1 = 221 -> (32, 221, 221)
        # after max pool => (32, 110, 110)
        self.drop1 = nn.Dropout(0.1)
        
        self.conv2 = nn.Conv2d(32, 64, 3)   # (110-3)/1 + 1 = 108 -> (64, 108, 108)
        # after max pool => (64, 54, 54)
        self.drop2 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv2d(64, 128, 2)  # (54-2)/1 + 1 = 53 -> (128, 53, 53)
        # after max pool => (128, 26, 26)
        self.drop3 = nn.Dropout(0.3)
        
        self.conv4 = nn.Conv2d(128, 256, 1)  # (26-1)/1 + 1 = 26 -> (256, 26, 26)
        # after max pool => (256, 13, 13)
        self.drop4 = nn.Dropout(0.4)
        
        self.dense1 = nn.Linear(256*13*13, 256*13)
        self.drop5 = nn.Dropout(0.5)
        
        self.dense2 = nn.Linear(256*13, 256*13)
        self.drop6 = nn.Dropout(0.6)
        
        self.dense3 = nn.Linear(256*13, 136)
                
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
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

