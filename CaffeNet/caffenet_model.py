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
import caffenet_settings as settings
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
    # tf.nn.zero_function: return the fraction of zeros in value. 0の断片を返す
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _filter_summary(x):
    '''
    filterの可視化サマリ
    '''
    x_input = tf.get_default_graph().as_graph_element(x)
    x_viz = tf.transpose(x_input, perm=[3, 0, 1, 2])
    #[:, :, :, : 0]
    #pool5_flat = tf.reshape(pool5, [FLAGS.batch_size, dim])
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.image_summary(tensor_name + '/filters', x_viz, max_images=24)

def inference(images):
    #'conv1 kernel 11x11, stride 4, output_map 55x55x64, af ReL'
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[11, 11, 3, 64], stddev=1e-1, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)
        _filter_summary(kernel)

    #'pool1 kernel 3x3, stride 2, output_map 27x27x64'
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

    #'norm1 kernel 5x5 stride 1, output_map 27x27x64'
    norm1 = tf.nn.local_response_normalization(pool1, 5, bias=1.0, alpha=2e-05, beta=0.75, name='norm1')

    #'conv2 kernel 5x5, stride 1, output_map 27x27x192, af ReL'
    with tf.variable_scope('conv2') as scope:
        kernel2 = _variable_with_weight_decay('weights', shape=[11, 11, 64, 192], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel2, [1, 4, 4, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)

    #'pool2 kernel 3x3, stride 2, output_map 13x13x256'
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    
    #'norm2 kernel 5x5, stride 1, output_map 13x13x256'
    norm2 = tf.nn.local_response_normalization(pool2, 5, bias=1.0, alpha=2e-05, beta=0.75, name='norm2')

    #'conv3 kernel 3x3, stride 1, output_map 13x13x384'
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 192, 384], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv3)

    #'conv4 kernel 3x3, stride 1, output_map 13x13x384'
    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 384, 384], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv4)

    #'conv5 kernel 3x3, stride 1, output_map 13x13x256'
    with tf.variable_scope('conv5') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 384, 256], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv5)


    #'pool5 kernel 3x3, stride 2, output_map 6x6x256'
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
    
    #'fc6 output_map 1x1x4096'
    with tf.variable_scope('fc6'):
        input_shape = pool5.get_shape()
        dim = 1
        for d in input_shape[1:].as_list():
            dim *= d
        pool5_flat = tf.reshape(pool5, [FLAGS.batch_size, dim])
        weights = _variable_with_weight_decay('weights', [dim, 4096], stddev=1/256.0, wd=0.04)
        biases = _variable_on_cpu('biases', 4096, tf.constant_initializer(0.0))
        fc6 = tf.nn.relu_layer(pool5_flat, weights, biases, name=scope.name)
        _activation_summary(fc6)

    #'fc6_dropout dropout 0.5'
    fc6_dropout = tf.nn.dropout(fc6, 0.5)

    #'fc7 output_map 1x1x4096'
    with tf.variable_scope('fc7'):
        input_shape = fc6_dropout.get_shape()
        inputs_fc7, dim = (fc6_dropout, int(input_shape[-1]))
        weights = _variable_with_weight_decay('weights', [4096, 4096], stddev=1/4096.0, wd=0.04)
        biases = _variable_on_cpu('biases', 4096, tf.constant_initializer(0.0))
        fc7 = tf.nn.relu_layer(inputs_fc7, weights, biases, name=scope.name)
        _activation_summary(fc7)

    #'fc7_dropout dropout 0.5'
    fc7_dropout = tf.nn.dropout(fc7, 0.5)

    #'fc8(softmax) output_map 1x1xNUM_CLASSES'
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [4096, NUM_CLASSES], stddev=1/4096.0, wd=0.04)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.nn.xw_plus_b(fc7_dropout, weights, biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, FLAGS.batch_size, 1), 1)
    concated = tf.concat(1, [indices, labels])
    # sparse_to_dense のクラス数は クラスラベルの最大値+1(クラス数) とすること
    dense_labels = tf.sparse_to_dense(
        concated,
        [FLAGS.batch_size, NUM_CLASSES],
        1.0,
        0.0
    )

    # target labelとの差を計算
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits,
        dense_labels,
        name='cross_entropy_per_example'
    )

    # computes the mean of elements across dimensions of a tensor
    # バッチの平均ロス
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    # 全部lossesに登録
    tf.add_to_collection('losses', cross_entropy_mean)

    # add all input tensors element wise.
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op
