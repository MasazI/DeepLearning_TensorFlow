# DeepLearning_TensorFlow
a python implementation of deeplearning using TensorFlow

## TensorFlow ##
http://tensorflow.org/

## Features

- Loading Data
- Createing mini-batch
- LogisticRegression
- MLP using tensorflow structure and tensorboard
- CNN_minit for new algorithm experiments with MNIST
- CNN_cminit for new algorithm experiments with Cluttered MNIST
- CNN_tiny(AlexNet 2010)
- CaffeNet(AlexNet 2012)
- BatchNormalization
- SpatialTransfomerNetwork
- SpatialTransfomerNetwork with AlexNet 2012 (pretrain and fine-tuning structure)

if you use pyenv and anaconda:
add /path/to/.pyenv/versions/anaconda-2.0.1/bin/ into PATH

## CNN
### CaffeNet
train  
cd CaffeNet and  
`python caffenet_train.py`

### BatchNormalization
train  
cd BatchNormalization and  
`python train.py`


## visualization  
`tensorboard --logdir /to/your/path/train --port=6006`



---

Copyright (c) 2015 Masahiro Imai
Released under the MIT license
