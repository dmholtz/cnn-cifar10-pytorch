"""
Trains a convolutional neural network to perform multi-class classification on
the CIFAR10 dataset. Built with PyTorch in python.

This training tool imports any neural network architectures, which is
definded in a separate class and imported as module.

The dataset is downloaded and the labeled data is split into training-
validation- and testing data. Moreover, data augmentation is performed, which
includes basic transformations such as horizontal flips or random rotations.

While training, validation loss is logged and every improvement is saved as
as trained model to disc.

@author: dmholtz
@version: 1.0
"""

import torch
import torch.nn as nn
import numpy as np
from numpy import genfromtxt
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import Utilities as util

# =============================================================================
# Initial definition of model architecture and hyperparamters
# =============================================================================

# choose the model architecture: the module which contains the model definition
package = 'model'
architecture = 'CNN6_FC2'

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

''' 
Data Augmentation

Defines a basic transformation pipeline for data augmentation. Transformations
may include random (horizontal) flips of pictures or small roatations.

The transformation pipeline also defines that input pictures are turned into
tensors and defines a normalization step to speed up gradient descent.
'''
transform = transforms.Compose([
    # data augmentation
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomGrayscale(0.1),
    transforms.RandomAffine(degrees = 10, translate = (0.1,0.1)),
    # convert data to a normalized torch.FloatTensor
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

print('Data augmentation and transformation pipeline:\n',transform)

# choose and download both training and test set
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)

pretrained_available = True

# obtain training indices that will be used for validation
# 1. get the size of the training set
num_train = len(train_data)

indices = None
if pretrained_available:
    indices = genfromtxt(architecture+'/indices.csv', delimiter = ',')
    indices = indices.astype(int)
    indices = list(indices)
else:
    # 2. create a list enumarating all the indices in the training set
    indices = list(range(num_train))
    # 3. shuffle indices: 
    np.random.shuffle(indices)
    np.savetxt(architecture+'/indices.csv', indices, delimiter=',')
    
indices = np.flip(indices)    
    
# 4. calculate the split index, which is a integer percentage of the whole
#   training set
splitIndex = int(validation_set_size * num_train)
# 5. Separate indices of training and validation data
train_idx, valid_idx = indices[splitIndex:], indices[:splitIndex]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)

# =============================================================================
# Visualizing a batch of trainig data
# =============================================================================

# optional
if showImages:
    util.showSample(train_loader, 10)
else:
    print('Start without previewing images...')

# =============================================================================
# Build up the model, define criterion optimizers
# =============================================================================

# Create a CNN according to the specifications above
model = cnn.Net()
print(model)

def loadPretrainedModel(loadOption):
    if loadOption:        
        model.load_state_dict(torch.load(architecture+'/model_cifar.pt'))
    else:
        print('Start training from scratch')
    return

# Loads the pre-trained model to continue training process
loadPretrainedModel(True)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()

# specify loss function: Cross Entropy Loss 
criterion = nn.CrossEntropyLoss()

# specify optimizer: stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay = reg)

# =============================================================================
# Train the model
# =============================================================================

valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # validate the model #
    ######################
    model.eval()
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
    
    # calculate average losses
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
    
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.3f} \tValidation Loss: {:.3f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.3f} --> {:.3f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), architecture+'/model_cifar a{}.pt'.format(epoch+48))
        torch.save(model.state_dict(), architecture+'/model_cifar.pt')
        valid_loss_min = valid_loss
        
    # update loss logger
    validationLossLogger = np.genfromtxt(
            architecture+'/validationLoss.csv', delimiter = ',')
    validationLossLogger = validationLossLogger.reshape(-1, 1)
    trainingLossLogger = np.genfromtxt(
            architecture+'/trainingLoss.csv', delimiter = ',')
    trainingLossLogger = trainingLossLogger.reshape(-1, 1)
    validationLossLogger = list(validationLossLogger)
    validationLossLogger.append(valid_loss)
    trainingLossLogger = list(trainingLossLogger)
    trainingLossLogger.append(train_loss)
    np.savetxt(architecture+'/validationLoss.csv',
               validationLossLogger, delimiter = ',')
    np.savetxt(architecture+'/trainingLoss.csv',
               trainingLossLogger, delimiter = ',')
        
