"""
Convolutional neural network for CIFAR10 dataset, built with PyTorch in python

@author: dmholtz
"""

suffix = ''

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
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

# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

    # obtain one batch of test images
    images, labels = data, target
    images.numpy()
    
    # move model inputs to cuda, if GPU available
    if train_on_gpu:
        images = images.cuda()
        
    # helper function to un-normalize and display an image
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
    
    # get sample outputs
    output = model(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())
    
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        imshow(images.cpu()[idx])
        ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                     color=("green" if preds[idx]==labels[idx].item() else "red"))
    
    plt.show(block = True)
    n = input('Press enter to continue or b to break: ')
    if n == 'b':
        break
        
