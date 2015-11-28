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
    
    hidden1_units = hidden_units[0]
    hidden2_units = hidden_units[1]

    with tf.name_scope('hidden1') as scope:
        weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units], stddev=1.0/math.sqrt(float(IMAGE_PIXELS))), name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    with tf.name_scope('hidden2') as scope:
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units], stddev=1.0/math.sqrt(float(hidden1_units))), name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    with tf.name_scope('softmax_linear') as scope:
        weights = tf.Variable(tf.truncated_normal([hidden2_units, NUM_CLASSES], stddev=1.0/math.sqrt(float(hidden2_units))), name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        logits = tf.matmul(hidden2, weights) + biases

    return logits



def test():
    pass

if __name__ == '__main__':
    test() 
