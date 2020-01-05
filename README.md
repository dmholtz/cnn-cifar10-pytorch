# cnn-cifar10-pytorch
Convolutional neural network for Cifar10 dataset, built with PyTorch  in python

Train and test several CNN models for cifar10 dataset.

## Model CNN3_FC2
* three convolutional layers
* two fully connected layers

## Model CNN4_FC2
* four convolutional layers
* two fully connected layers

## Model CNN5_FC2
poor performance compared to other models
* five convolutional layers
* two fully connected layers

## Model CNN6_FC2
best performance among all models (84,4% accuracy on the test set)
* six convolutional layers
* two fully connected layers

Training process
* data augmentation: affine transformations (rotations, translations), random b/w images
* regularization and dropout

## System information
Built with python 3.7.4 (Anaconda). Requires PyTorch and MatplotLib.
