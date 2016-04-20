# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import sys
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf

import cv2

import model

#import cnn_tiny_settings as settings
#FLAGS = settings.FLAGS

NUM_CLASSES = 5
IMAGE_SIZE = 24
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

# 学習時のbatch size
BATCH_SIZE = 1

def _variable_with_weight_decay(name, shape, stddev, wd):
    '''
    重み減衰を利用した変数の初期化
    '''
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _variable_on_cpu(name, shape, initializer):
    '''
    CPUメモリに変数をストアする
    '''
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

if __name__ == '__main__':
    test_image = []
    for i in range(1, len(sys.argv)):
        image = cv2.imread(sys.argv[i])
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        b, g, r = cv2.split(image)       # get b,g,r
        rgb_image = cv2.merge([r, g, b])     # switch it to rgb
        test_image.append(rgb_image.flatten().astype(np.float32))
    test_image = np.asarray(test_image)
    images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
    
    logits = model.inference(images_placeholder)
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    # restore trained model
    ckpt = tf.train.get_checkpoint_state('train')
    print(ckpt.model_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
        print('No checkpoint file found.')
        quit()

    # predict
    print('the num of test images: %d' % (len(test_image)))
    for i in range(len(test_image)):
        start_time = time.time()
        softmax = logits.eval(feed_dict={images_placeholder: [test_image[i]]}) [0]
        print(softmax)
        pred = np.argmax(softmax)
        duration = time.time() - start_time
        print('category: %i, duration: %f (sec)' % (pred, duration))

