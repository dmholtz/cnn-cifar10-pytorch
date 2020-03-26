"""
Tests the trained neural network on custom images. Images must be stored where
the variable filename refers to.

Please note: The Convolutional Neural Network is trained on three channel RGB
images. In case a custom image does not meet this requirement, the image must
be manually converted in advanced.

Images, which are not 32x32 pixels in size, will be cropped symmetrically and
then scaled to the (32x32) pixels. This version will be forward passed through
the neural network. Morover, this script preserves an higher resolution
input cropped input image (256x256), which will be displayed in the console.

@author: dmholtz
"""

import torch
import Utilities as util
import CifarResources as cifar

# =============================================================================
# Initial definition of model architecture and hyperparamters
# =============================================================================

# choose the model architecture: the module which contains the model definition
package = 'model'
architecture = 'CNN6_FC2'

# relative location (path+filename) of the custom image
filename = 'images/testimage.jpg'

# =============================================================================
# Setup and user feedback
# =============================================================================

# try to import the model architecture definition module    
cnn = util.importModelArchitecture(package, architecture)

# check if CUDA is available on this computer
train_on_gpu = torch.cuda.is_available()
print('Cuda available?: ' + ('Yes' if train_on_gpu else 'No'))

# =============================================================================
# Load the model
# =============================================================================

# Create a CNN according to the specification in CNN3_FC2
model = cnn.Net()
print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()
        
model.load_state_dict(torch.load(architecture+'/model_cifar.pt'))

from PIL import Image
import torchvision.transforms.functional as TF

rawImage = Image.open(filename)

width, height = rawImage.size
squaredSize = min(width, height)

horizontalCrop = (width - height) // 2 if (width > height) else 0
verticalCrop = (height - width) // 2 if (height > width) else 0

# crop(upperLeftHandX, uperLeftHandY, lowerRightHandX, lowerRightHandY)
rawImage = rawImage.crop((horizontalCrop, verticalCrop, squaredSize+horizontalCrop, squaredSize+verticalCrop))

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
