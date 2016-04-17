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


def _activation_summary(x):
    '''
    可視化用のサマリを作成
    '''
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def inference(images, re_images, keep_conv, keep_hidden):
    '''
    アーキテクチャの定義、グラフのビルド
    '''
    # spatial transformer
    with tf.variable_scope('spatial_transformer') as scope:
        #reshape = tf.reshape(images, [-1, ])
        weights_loc1 = _variable_with_weight_decay(
            'weights_loc1',
            shape=[1600, 20],
            stddev=0.01,
            wd=0.0
        )
        biases_loc1 = _variable_on_cpu('biases_loc1', [20], tf.constant_initializer(0.1))
        weights_loc2 = _variable_with_weight_decay(
            'weights_loc2',
            shape=[20, 6],
            stddev=0.01,
            wd=0.0
        )
        initial = np.array([[1., 0, 0], [0, 1., 0]]) # Use identity transformation as starting point
        initial = initial.astype('float32')
        initial = initial.flatten()
        biases_loc2 = tf.Variable(initial_value=initial, name='biases_loc2')
        
        # define the two layer localisation network
        h_fc_loc1 = tf.nn.tanh(tf.matmul(images, weights_loc1) + biases_loc1)
        # We can add dropout for regularizing and to reduce overfitting like so:
        h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_conv)
        # Second layer
        h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, weights_loc2) + biases_loc2)
       
        # Transformer layer 
        hidden_trans = spatial.transformer(re_images, h_fc_loc2, downsample_factor=1)
        _activation_summary(hidden_trans)

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[3, 3, 1, 16],
            stddev=0.01,
            wd=0.0 # not use weight decay
        )
        conv = tf.nn.conv2d(hidden_trans, kernel, [1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.relu(conv, name=scope.name)
        _activation_summary(conv1)

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[3, 3, 16, 16],
            stddev=0.01,
            wd=0.0 # not use weight decay
        )
        conv = tf.nn.conv2d(conv1, kernel, [1, 2, 2, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)
    
    # local3 fc
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(conv2, [-1, 10*10*16])
        reshape = tf.nn.dropout(reshape, keep_conv) 
        weights = _variable_with_weight_decay(
            'weights',
            shape=[10*10*16, 1024],
            stddev=0.01,
            wd=0.04
        )
        biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.add(tf.matmul(reshape, weights), biases, name=scope.name))
        _activation_summary(local3)

    dropout3 = tf.nn.dropout(local3, keep_hidden)

    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay(
            'weights',
            [1024, NUM_CLASSES],
            stddev=0.01,
            wd=0.0
        )
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(dropout3, weights), biases, name=scope.name)
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
