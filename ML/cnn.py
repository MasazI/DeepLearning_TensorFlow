#encoding:utf-8
import tensorflow as tf
import logistic_regression

class CNN():
    def __init__()

    def wegiht_variable(shape):
        # 初期値を生成したweight
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        # 初期値を指定したbias
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 2], strides=[1, 2, 2, 1], padding='SAME')
