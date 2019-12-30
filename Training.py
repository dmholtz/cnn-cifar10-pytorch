"""
Convolutional neural network for CIFAR10 dataset, built with PyTorch in python

@author: dmholtz
"""

from model import CNN3_FC2 as cnn

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

# check if CUDA is available on this computer
train_on_gpu = torch.cuda.is_available()
print('Cuda available?: ', train_on_gpu)

# =============================================================================
# Initial definition of some hyperparamters
# =============================================================================

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation set
validation_set_size = 0.2
# learning rate
learning_rate = 0.01
# number of epochs to train the model
n_epochs = 50

# =============================================================================
# Obtaining and preprocessing data
# =============================================================================

# defining transformation pipeline:
# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# choose and download both training and test set
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)

# obtain training indices that will be used for validation
# 1. get the size of the training set
num_train = len(train_data)
# 2. create a list enumarating all the indices in the training set
indices = list(range(num_train))
# 3. shuffle indices: 
np.random.shuffle(indices)
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

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# =============================================================================
# Visualizing a batch of trainig data
# =============================================================================

def showSample(show):
    if show:
        import matplotlib.pyplot as plt

        # helper function to un-normalize and display an image
        def imshow(img):
            img = img / 2 + 0.5  # unnormalize
            plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
        
        # obtain one batch of training images
        dataiter = iter(train_loader)
        images, labels = dataiter.next()
        images = images.numpy() # convert images to numpy for display
        
        # plot the images in the batch, along with the corresponding labels
        fig = plt.figure(figsize=(25, 4))
        # display 20 images
        for idx in np.arange(20):
            ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
            imshow(images[idx])
            ax.set_title(classes[labels[idx]])
    
    else:
        print('-> No preview of a sample of training data')
            
# define whether a sample should be shown or not
showSample(False)

# =============================================================================
# Build up the model, define criterion optimizers
# =============================================================================

# Create a CNN according to the specification in CNN3_FC2
model = cnn.Net()
model.load_state_dict(torch.load('model_cifar.pt'))
print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()

# specify loss function: Cross Entropy Loss 
criterion = nn.CrossEntropyLoss()

# specify optimizer: stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

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
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss
        

print ('finished so far')