# encoding: utf-8

# general
import os
import re
import sys

# tensorflow
import tensorflow as tf

import numpy as np
import math

# settings
import settings
FLAGS = settings.FLAGS

NUM_CLASSES = FLAGS.num_classes
LEARNING_RATE_DECAY_FACTOR = FLAGS.learning_rate_decay_factor
INITIAL_LEARNING_RATE = FLAGS.learning_rate

# multiple GPU's prefix
TOWER_NAME = FLAGS.tower_name


def _variable_with_weight_decay(name, shape, stddev, wd):
    '''
    重み減衰を利用した変数の初期化
    '''
    var = _variable_on_gpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _variable_on_cpu(name, shape, initializer):
    '''
    CPUメモリに変数をストアする
    '''
    #with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_on_gpu(name, shape, initializer):
    '''
    GPUメモリに変数をストアする
    '''
    #with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _activation_summary(x):
    '''
    可視化用のサマリを作成
    '''
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def inference(images, keep_conv, keep_hidden):
    '''
    アーキテクチャの定義、グラフのビルド
    '''
    print "="*100
    print "input"
    print images.get_shape()
    print "="*100

    # course1 input image 11x11conv, 4stride -> 155, 115, 96
    with tf.variable_scope('coarse1') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[11, 11, 3, 96],
            stddev=0.01,
            wd=0.0 # not use weight decay
        )
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='VALID')
        biases = _variable_on_gpu('biases', [96], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        coarse1_conv = tf.nn.relu(bias, name=scope.name)
        _activation_summary(coarse1_conv)

    print "="*100
    print "coarse1 conv"
    print coarse1_conv.get_shape()
    print "="*100

    # coarse1 kernel 2x2, stride 1, output_map -> 154, 114, 96
    coarse1 = tf.nn.max_pool(coarse1_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')


    print "="*100
    print "coarse1"
    print coarse1.get_shape()
    print "="*100
 
    # coarse2
    with tf.variable_scope('coarse2') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[5, 5, 96, 256],
            stddev=0.01,
            wd=0.0 # not use weight decay
        )
        conv = tf.nn.conv2d(coarse1, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variable_on_gpu('biases', [256], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        coarse2_conv = tf.nn.relu(bias, name=scope.name)
        _activation_summary(coarse2_conv)

    print "="*100
    print "coarse2 conv"
    print coarse2_conv.get_shape()
    print "="*100

    # pool1 kernel 2x2, stride 1, output_map ??x??x??
    coarse2 = tf.nn.max_pool(coarse2_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    print "="*100
    print "coarse2"
    print coarse2.get_shape()
    print "="*100

    
    # coarse3
    with tf.variable_scope('coarse3') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[3, 3, 256, 384],
            stddev=0.01,
            wd=0.0 # not use weight decay
        )
        conv = tf.nn.conv2d(coarse2, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variable_on_gpu('biases', [384], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        coarse3 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(coarse3)

    print "="*100
    print "coarse3"
    print coarse3.get_shape()
    print "="*100

    # coarse4
    with tf.variable_scope('coarse4') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[3, 3, 384, 384],
            stddev=0.01,
            wd=0.0 # not use weight decay
        )
        conv = tf.nn.conv2d(coarse3, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variable_on_gpu('biases', [384], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        coarse4 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(coarse4)

    print "="*100
    print "coarse4"
    print coarse4.get_shape()
    print "="*100

    # coarse5
    with tf.variable_scope('coarse5') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[3, 3, 384, 256],
            stddev=0.01,
            wd=0.0 # not use weight decay
        )
        conv = tf.nn.conv2d(coarse4, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variable_on_gpu('biases', [256], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        coarse5 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(coarse5)

    print "="*100
    print "coarse5 to full layer"
    print coarse5.get_shape()
    print "="*100

    # coarse6 full layer
    with tf.variable_scope('coarse6'):
        # input_shape = coarse5.get_shape()
        coarse5_flat = tf.reshape(coarse5, [-1, 6*10*256])
        weights = _variable_with_weight_decay(
            'weights',
            [6*10*256, 4096],
            stddev=0.01,
            wd=0.04)
        biases = _variable_on_gpu('biases', 4096, tf.constant_initializer(0.1))
        coarse6 = tf.nn.relu_layer(coarse5_flat, weights, biases, name=scope.name)
        _activation_summary(coarse6)

    print "="*100
    print "coarse6 to full layer"
    print coarse6.get_shape()
    print "="*100

    # course7 full layer
    with tf.variable_scope('coarse7'):
        # input_shape = coarse5.get_shape()
        weights = _variable_with_weight_decay(
            'weights',
            [4096, 4070],
            stddev=0.01,
            wd=0.04)
        biases = _variable_on_gpu('biases', 4070, tf.constant_initializer(0.1))
        coarse7 = tf.nn.relu_layer(coarse6, weights, biases, name=scope.name)
        _activation_summary(coarse7)

    print "="*100
    print "coarse6 to full layer"
    print coarse7.get_shape()
    print "="*100

    coarse7_output = tf.reshape(coarse7, [-1, 55, 74, 1])

    print "="*100
    print "coarse7"
    print coarse7_output.get_shape()
    print "="*100

    # fine1 input image 9x9conv, 2stride -> , , 
    with tf.variable_scope('fine1') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[9, 9, 3, 63],
            stddev=0.01,
            wd=0.0 # not use weight decay
        )
        conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='VALID')
        biases = _variable_on_gpu('biases', [63], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        fine1_conv = tf.nn.relu(bias, name=scope.name)
        _activation_summary(fine1_conv)

    print "="*100
    print "fine1 conv"
    print fine1_conv.get_shape()
    print "="*100

    # fine1 kernel 2x2, stride 1, output_map -> , , 
    fine1 = tf.nn.max_pool(fine1_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    print "="*100
    print "fine1"
    print fine1.get_shape()
    print "="*100
  

    # concatenate fine1 and coarse7_output
    fine2 = tf.concat(3, [fine1, coarse7_output])

    print "="*100
    print "fine2"
    print fine2.get_shape()
    print "="*100
 

    with tf.variable_scope('fine3') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[5, 5, 64, 64],
            stddev=0.01,
            wd=0.0 # not use weight decay
        )
        conv = tf.nn.conv2d(fine2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_gpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        fine3 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(fine3)

    print "="*100
    print "fine3"
    print fine3.get_shape()
    print "="*100
 
    with tf.variable_scope('fine4') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[5, 5, 64, 1],
            stddev=0.01,
            wd=0.0 # not use weight decay
        )
        conv = tf.nn.conv2d(fine2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_gpu('biases', [1], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        fine4 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(fine4)

    print "="*100
    print "fine4"
    print fine4.get_shape()
    print "="*100
 
    return coarse7_output, fine4

def loss(logits, depths, invalid_depths):
    logits_flat = tf.reshape(logits, [-1, 55*74])
    depths_flat = tf.reshape(depths, [-1, 55*74])
    invalid_depths_flat = tf.reshape(depths, [-1, 55*74])
    
    # ignore 0 depths in ground truth.
    predict = tf.mul(logits_flat, invalid_depths_flat)
    target = tf.mul(depths_flat, invalid_depths_flat)
    d = tf.sub(predict, target)

    # term1
    square_d = tf.square(d)
    sum_square_d = tf.reduce_sum(square_d, 1)

    # term 2
    sum_d = tf.reduce_sum(d, 1)
    sqare_sum_d = tf.square(sum_d)

    # term1 - lambda * term2
    cost = tf.reduce_mean(sum_square_d / 55*74 - FLAGS.si_lambda*sqare_sum_d / math.pow(55*74, 2))
    tf.add_to_collection('losses', cost)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op
