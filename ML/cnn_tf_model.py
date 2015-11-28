#encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow.python.platform
import tensorflow as tf

# category num
NUM_CLASSES = 10

# static image size of mnist
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def inference(images, hidden_units):
    '''
    build model

    arguments:
        images: Images placeholder
        hidden_units: Array. size of hidden layer.

        ex) [10, 20]
        size of the first hidden layer is 10
        size of the second hidden layer is 20 
    
    returns:
       softmax_linear: output tensor with computed data called logits. 
    '''
    
    
