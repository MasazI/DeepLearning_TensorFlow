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

def inference(images):
    '''
    アーキテクチャの定義、グラフのビルド
    '''
    # 学習時にネットワークに入力していた画像サイズに揃える
    x_image = tf.reshape(images, [-1, 24, 24, 3])

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[5, 5, 3, 64],
            stddev=1e-4,
            wd=0.0 # not use weight decay
        )
        conv = tf.nn.conv2d(x_image, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(
        conv1,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='pool1'
    )

    # norm1
    norm1 = tf.nn.lrn(
        pool1,
        4,
        bias=1.0,
        alpha=0.001/9.0,
        beta=0.75,
        name='norm1'
    )

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[5, 5, 64, 64],
            stddev=1e-4,
            wd=0.0 # not use weight decay
        )
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)

    # norm2
    norm2 = tf.nn.lrn(
        conv2,
        4,
        bias=1.0,
        alpha=0.001/9.0,
        beta=0.75,
        name='norm2'
    )

    # pool2
    pool2 = tf.nn.max_pool(
        norm2,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='pool2'
    )

    # local3 局所正規化
    with tf.variable_scope('local3') as scope:
        dim = 1
        for d in pool2.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool2, [BATCH_SIZE, dim])

        weights = _variable_with_weight_decay(
            'weights',
            shape=[dim, 384],
            stddev=0.04,
            wd=0.004
        )
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu_layer(reshape, weights, biases, name=scope.name)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay(
            'weights',
            shape=[384, 192],
            stddev=0.04,
            wd=0.004
        )
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu_layer(local3, weights, biases, name=scope.name)

    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay(
            'weights',
            [192, NUM_CLASSES],
            stddev=1/192.0,
            wd=0.0
        )
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.nn.xw_plus_b(local4, weights, biases, name=scope.name)

    return softmax_linear

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
    
    logits = inference(images_placeholder)
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    ckpt = tf.train.get_checkpoint_state('train')
    print(ckpt.model_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
        print('No checkpoint file found.')
        quit()

    for i in range(len(test_image)):
        print(logits.eval(feed_dict={images_placeholder: [test_image[i]]}))
        pred = np.argmax(logits.eval(feed_dict={images_placeholder: [test_image[i]]}) [0])
        print(pred)


