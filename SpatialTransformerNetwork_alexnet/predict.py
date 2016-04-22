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


# settings
import settings
FLAGS = settings.FLAGS
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
    for path in range(1, len(sys.argv)):
        img = cv2.imread(sys.argv[path])
        h, w, c = np.shape(img)
        resize_height = FLAGS.image_height
        resize_width = FLAGS.image_width
        crop_size = FLAGS.crop_size
        img = cv2.resize(img, (resize_height, resize_width))
        img = img.astype(np.float32)

        # Fine Tuningの際は注意すること
        # img -= self.mean
        
        h, w, c = img.shape
        crop_height, crop_width = ((h-crop_size)/2, (w-crop_size)/2)
        img = img[crop_height:crop_height+crop_size, crop_width:crop_width+crop_size, :]
        img = img[None, ...]
        test_image.append(img)
    test_image = np.asarray(test_image)
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    keep_conv = tf.placeholder(tf.float32)
    keep_hidden = tf.placeholder(tf.float32)

    logits = model.inference(images, keep_conv, keep_hidden)
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    # restore trained model
    ckpt = tf.train.get_checkpoint_state('trained_model')
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
        softmax = logits.eval(feed_dict={images: test_image[i], keep_conv: 1.0, keep_hidden: 1.0}) [0]
        print(softmax)
        pred = np.argmax(softmax)
        duration = time.time() - start_time
        print('category: %i, duration: %f (sec)' % (pred, duration))

