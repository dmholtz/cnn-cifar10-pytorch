"""
Convolutional neural network with five convolutional layer and two fully-
connected layers afterwards

@author: dmholtz
"""

import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1. convolutional layer
        # sees 32x32x3 image tensor, i.e 32x32 RGB pixel image
        # outputs 32 filtered images, kernel-size is 3
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # 2. convolutional layer
        # sees 16x16x32 tensor (2x2 MaxPooling layer beforehand)
        # outputs 32 filtered images, kernel-size is 3
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 3. convolutional layer
        # sees 8x8x32 tensor (2x2 MaxPooling layer beforehand)
        # outputs 64 filtered images, kernel-size is 3
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Definition of the MaxPooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # 1. fully-connected layer
        # Input is a flattened 4*4*64 dimensional vector
        # Output is 500 dimensional vector
        self.fc1 = nn.Linear(128 * 4 * 4, 500)
        # 2. fully-connected layer
        # 500 nodes into 10 nodes (i.e. the number of classes)
        self.fc2 = nn.Linear(500, 10)
        
        # defintion of dropout (dropout probability 25%)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Pass data through a sequence of 3 convolutional layers
        # Firstly, filters are applied -> increases the depth
        # Secondly, Relu activation function is applied
        # Finally, MaxPooling layer decreases width and height
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        #x = self.pool(x)
        
        x = self.pool(F.relu(self.conv4(x)))
        
        # flatten output of third convolutional layer into a vector
        # this vector is passed through the fully-connected nn
        x = x.view(-1, 128 * 4 * 4)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, without relu activation function
        x = self.fc2(x)
        return x

