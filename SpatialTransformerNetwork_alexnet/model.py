# encoding: utf-8

# general
import os
import re
import sys

# tensorflow
import tensorflow as tf

import spatial
import numpy as np

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
    # spatial transformer
    reshape = tf.reshape(images, [-1, 224*224*3])
    with tf.variable_scope('spatial_transformer') as scope:
        # localisation net
        weights_loc1 = _variable_with_weight_decay(
            'weights_loc1',
            shape=[224*224*3, 20],
            stddev=0.01,
            wd=0.0
        )
        biases_loc1 = _variable_on_gpu('biases_loc1', [20], tf.constant_initializer(0.1))
        # output 6 dimentional vector to compute sampling grid
        weights_loc2 = _variable_with_weight_decay(
            'weights_loc2',
            shape=[20, 6],
            stddev=0.01,
            wd=0.0
        )
        # Use identity transformation as starting point
        initial = np.array([[1., 0, 0], [0, 1., 0]])
        initial = initial.astype('float32')
        initial = initial.flatten()
        biases_loc2 = tf.Variable(initial_value=initial, name='biases_loc2')
        
        # define the two layer localisation network
        h_fc_loc1 = tf.nn.tanh(tf.matmul(reshape, weights_loc1) + biases_loc1)
        # We can add dropout for regularizing and to reduce overfitting like so:
        h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_conv)
        # Second layer
        h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, weights_loc2) + biases_loc2)
       
        # Transformer layer 
        hidden_trans = spatial.transformer(images, h_fc_loc2, downsample_factor=1)
        _activation_summary(hidden_trans)

    print "="*100
    print images.get_shape()
    print "="*100
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[11, 11, 3, 64],
            stddev=0.01,
            wd=0.0 # not use weight decay
        )
        conv = tf.nn.conv2d(hidden_trans, kernel, [1, 4, 4, 1], padding='SAME')
        biases = _variable_on_gpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)

    print "="*100
    print conv1.get_shape()
    print "="*100
 

    # pool1 kernel 3x3, stride 2, output_map 27x27x64
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

    # norm1 kernel 5x5 stride 1, output_map 27x27x64
    norm1 = tf.nn.local_response_normalization(pool1, 5, bias=1.0, alpha=2e-05, beta=0.75, name='norm1')

    print "="*100
    print norm1.get_shape()
    print "="*100
 

    # conv2 kernel 5x5, stride 1, output_map 27x27x192, af ReLu
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[5, 5, 64, 192],
            stddev=0.01,
            wd=0.0 # not use weight decay
        )
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_gpu('biases', [192], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)

    print "="*100
    print conv2.get_shape()
    print "="*100
 

    # pool2 kernel 3x3, stride 2, output_map 13x13x256
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    
    # norm2 kernel 5x5, stride 1, output_map 13x13x256
    norm2 = tf.nn.local_response_normalization(pool2, 5, bias=1.0, alpha=2e-05, beta=0.75, name='norm2')

    # conv3 kernel 3x3, stride 1, output_map 13x13x384
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[3, 3, 192, 384],
            stddev=1e-4,
            wd=0.0)
        conv = tf.nn.conv2d(norm2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_gpu('biases', [384], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv3)

    # conv4 kernel 3x3, stride 1, output_map 13x13x384
    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[3, 3, 384, 384],
            stddev=1e-4,
            wd=0.0)
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_gpu('biases', [384], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv4)

    # conv5 kernel 3x3, stride 1, output_map 13x13x256
    with tf.variable_scope('conv5') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[3, 3, 384, 256],
            stddev=1e-4,
            wd=0.0)
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_gpu('biases', [256], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv5)

    print "="*100
    print conv5.get_shape()
    print "="*100
 

    # pool5 kernel 3x3, stride 2, output_map 6x6x256
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
    
    # fc6 output_map 1x1x4096
    with tf.variable_scope('fc6'):
        input_shape = pool5.get_shape()
        pool5_flat = tf.reshape(pool5, [-1, 6*6*256])
        weights = _variable_with_weight_decay(
            'weights',
            [6*6*256, 4096],
            stddev=0.01,
            wd=0.04)
        biases = _variable_on_gpu('biases', 4096, tf.constant_initializer(0.1))
        fc6 = tf.nn.relu_layer(pool5_flat, weights, biases, name=scope.name)
        _activation_summary(fc6)

    # fc6_dropout dropout
    fc6_dropout = tf.nn.dropout(fc6, keep_hidden)

    # fc7 output_map 1x1x4096
    with tf.variable_scope('fc7'):
        input_shape = fc6_dropout.get_shape()
        inputs_fc7, dim = (fc6_dropout, int(input_shape[-1]))
        weights = _variable_with_weight_decay(
            'weights',
            [4096, 4096],
            stddev=0.01,
            wd=0.04)
        biases = _variable_on_gpu('biases', 4096, tf.constant_initializer(0.1))
        fc7 = tf.nn.relu_layer(inputs_fc7, weights, biases, name=scope.name)
        _activation_summary(fc7)

    # fc7_dropout dropout
    fc7_dropout = tf.nn.dropout(fc7, keep_hidden)

    # fc8(softmax) output_map 1x1xNUM_CLASSES
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay(
            'weights',
            [4096, NUM_CLASSES],
            stddev=0.01,
            wd=0.04)
        biases = _variable_on_gpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        softmax_linear = tf.nn.xw_plus_b(fc7_dropout, weights, biases, name=scope.name)
        _activation_summary(softmax_linear)
    
    return softmax_linear


def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits,
        labels,
        name='cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op
