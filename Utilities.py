# -*- coding: utf-8 -*-
"""
Utilities for training and testing neural networks

By dmholtz
"""
import importlib
import numpy as np
import matplotlib.pyplot as plt
import CifarResources as rsc

def importModelArchitecture(package, module):
    try:
        cnn = importlib.import_module(package+'.'+module)
        #cnn = importlib.import_module(architecture, 'model')
        #from model import module as cnn # import the desired module
        print('Successfully imported {} from package {}.'.format(
            module, package))
        return cnn
    except ImportError:
        print('Importing failed.')
        
def loadPretrainedModel(load = True):
    if load:
        return
    else:
        return
    
# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
        
def showSample(dataloader, numberOfImages = 20):  
    # obtain one batch of training images
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    images = images.numpy() # convert images to numpy for display
    
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    # display images
    for idx in np.arange(numberOfImages):
        ax = fig.add_subplot(2, numberOfImages/2, idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(rsc.classes[labels[idx]])
        
def saveLossToLogfile(file, loss):
    lossLogger = np.genfromtxt(file, delimiter = ',')
    lossLogger = lossLogger.reshape(-1, 1)
    lossLogger = list(lossLogger)
    lossLogger.append(loss)
    np.savetxt(file, lossLogger, delimiter = ',')
    