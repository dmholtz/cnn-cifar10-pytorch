"""
Convolutional neural network for CIFAR10 dataset, built with PyTorch in python

@author: dmholtz
"""

suffix = ''

import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import torch.optim as optim
import Utilities as util
import CifarResources as cifar

# =============================================================================
# Initial definition of model architecture and hyperparamters
# =============================================================================

# choose the model architecture: the module which contains the model definition
package = 'model'
architecture = 'CNN6_FC2'

filename = 'images/4.jpg'

# learning rate
learning_rate = 0.01
# regularization parameter
reg = 0.005
# percentage of training set to use as validation set
validation_set_size = 0.2
# number of epochs to train the model
n_epochs = 50

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20

# show image sample before training
showImages = True

# =============================================================================
# Setup and user feedback
# =============================================================================

# try to import the model architecture definition module    
cnn = util.importModelArchitecture(package, architecture)

# check if CUDA is available on this computer
train_on_gpu = torch.cuda.is_available()
print('Cuda available?: ' + ('Yes' if train_on_gpu else 'No'))

# =============================================================================
# Obtaining and preprocessing data
# =============================================================================

# defining transformation pipeline:
# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)

# obtain training indices that will be used for validation
# 1. get the size of the training set

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# =============================================================================
# Build up the model, define criterion optimizers
# =============================================================================

# Create a CNN according to the specification in CNN3_FC2
model = cnn.Net()
print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()

# specify loss function: Cross Entropy Loss 
criterion = nn.CrossEntropyLoss()

# specify optimizer: stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        
model.load_state_dict(torch.load(architecture+'/model_cifar'+suffix+'.pt'))

from PIL import Image
import torchvision.transforms.functional as TF

rawImage = Image.open(filename)

width, height = rawImage.size
squaredSize = min(width, height)
horizontalCrop = (width - height) // 2 if (width > height) else 0
verticalCrop = (height - width) // 2 if (height > width) else 0
leftCrop, topCrop = horizontalCrop, verticalCrop
rightCrop = squaredSize - leftCrop
bottomCrop = squaredSize - topCrop

rawImage = rawImage.crop((leftCrop, topCrop, rightCrop, bottomCrop))
rawImage = rawImage.resize((256,256))
rawImageData = TF.to_tensor(rawImage)
rawImageData.unsqueeze_(0)

image = rawImage.resize((32,32))

imageData = TF.to_tensor(image)
imageData.unsqueeze_(0)

model.eval()

# pass image data through the model
output = model(imageData)

# retrieve the class scores and the prediction
classScores, classPrediction = torch.topk(output, 10)
# process data and return probability distribution
classPrediction = classPrediction.detach().numpy().reshape(-1)
classScores = classScores.detach().numpy().reshape(-1)
probabilityDistribution = util.softmax(classScores)

predictedClass = cifar.classes[classPrediction[0]]

util.showSingleImage(rawImageData.numpy(), 'Input Image')

print('The image contains: {}'.format(predictedClass))
print('Probability distribution for all classes:')
for prob, pred in zip(probabilityDistribution, classPrediction):
    print('{:>6.2f}%:\t{}'.format(prob*100, cifar.classes[pred]))
    
util.showSingleImage(imageData.numpy(), predictedClass)
