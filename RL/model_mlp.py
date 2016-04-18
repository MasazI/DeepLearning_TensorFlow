# encoding: utf-8

# general
import re

# tensorflow
import tensorflow as tf

# data
#import load

# settings
import settings
FLAGS = settings.FLAGS

NUM_CLASSES = FLAGS.num_classes
LEARNING_RATE_DECAY_FACTOR = FLAGS.learning_rate_decay_factor
INITIAL_LEARNING_RATE = FLAGS.learning_rate

# multiple GPU's prefix
TOWER_NAME = FLAGS.tower_name

def weight_variable(shape):
    '''
    重み変数の生成
    '''
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def batch_normalization(shape, input):
    '''
    BatchNormalization
    '''
    eps = 1e-5
    gamma = weight_variable([shape])
    beta = weight_variable([shape])
    mean, variance = tf.nn.moments(input, [0])
    return gamma * (input - mean) / tf.sqrt(variance + eps) + beta

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


def inference(inputs, name):
    '''
    アーキテクチャの定義、グラフのビルド
    '''
    # layer1
    layer1_name = 'fc1_' + name
    with tf.variable_scope(layer1_name) as scope:
        weights = _variable_with_weight_decay(
            'weights',
            shape=[9, 12],
            stddev=0.04,
            wd=0.004
        )
        biases = _variable_on_cpu('biases', [12], tf.constant_initializer(0.1))
        #bn1 = batch_normalization(4, tf.matmul(inputs, weights))
        #local1 = tf.nn.relu(bn1)
        #inner_product = tf.matmul(inputs, weights)
        local1 = tf.nn.relu(tf.add(tf.matmul(inputs, weights), biases))
        #local1 = tf.nn.relu_layer(inputs, weights, biases, name=scope.name)
        #_activation_summary(local1)
    # softmax
    layer2_name = 'fc2_' + name
    with tf.variable_scope(layer2_name) as scope:
        weights = _variable_with_weight_decay(
            'weights',
            [12, NUM_CLASSES],
            stddev=0.04,
            wd=0.0
        )
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        linear = tf.nn.xw_plus_b(local1, weights, biases, name=scope.name)
        #_activation_summary(linear)

    return linear


def debug(data):
    return data

def loss(logits, targets, name):
    cost = tf.reduce_sum(tf.pow(logits - targets, 2))
    cost_mean_name = 'cost_' + name
    cost_mean = tf.reduce_mean(cost, name=cost_mean_name)
    losses_name = 'losses_' + name
    tf.add_to_collection(losses_name, cost_mean)

    total_name = 'total_loss_' + name
    return tf.add_n(tf.get_collection(losses_name), name=total_name)


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op
