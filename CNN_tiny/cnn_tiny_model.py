# encoding: utf-8

# general
import os
import re
import sys

# tensorflow
import tensorflow as tf

# data
import data

# inputs
import data_inputs

# settings
import cnn_tiny_settings as settings
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


def inference(images):
    '''
    アーキテクチャの定義、グラフのビルド
    '''
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[5, 5, 3, 64],
            stddev=1e-4,
            wd=0.0 # not use weight decay
        )
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu6(bias, name=scope.name)
        _activation_summary(conv1)

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
        _activation_summary(conv2)
    
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
        reshape = tf.reshape(pool2, [FLAGS.batch_size, dim])

        weights = _variable_with_weight_decay(
            'weights',
            shape=[dim, 384],
            stddev=1.0/dim,
            wd=0.04
        )
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu_layer(reshape, weights, biases, name=scope.name)
        _activation_summary(local3)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay(
            'weights',
            shape=[384, 192],
            stddev=1/384.0,
            wd=0.04
        )
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu_layer(local3, weights, biases, name=scope.name)
        _activation_summary(local4)

    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay(
            'weights',
            [192, NUM_CLASSES],
            stddev=1/192.0,
            wd=0.04
        )
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.nn.xw_plus_b(local4, weights, biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    #sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 1])
    #indices = tf.reshape(tf.range(0, FLAGS.batch_size), [FLAGS.batch_size, 1])
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, FLAGS.batch_size, 1), 1)
    #concated = tf.concat(1, [indices, sparse_labels])
    concated = tf.concat(1, [indices, labels])
    # sparse_to_dense のクラス数は クラスラベルの最大値+1 とすること
    dense_labels = tf.sparse_to_dense(
        concated,
        [FLAGS.batch_size, NUM_CLASSES],
        1.0,
        0.0
    )

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits,
        dense_labels,
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
