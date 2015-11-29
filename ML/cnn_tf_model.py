#encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow.python.platform
import tensorflow as tf

# category num
NUM_CLASSES = 10

# static image size of mnist
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# inference build model for feed-forward.
def inference(images, hidden_units):
    '''
    build model

    arguments:
        images: Images placeholder
        hidden_units: Array. size of hidden layer.

        ex) [10, 20]
        size of the first hidden layer is 10
        size of the second hidden layer is 20 
    
    returns:
       softmax_linear: output tensor with computed data called logits. 
    '''
    
    hidden1_units = hidden_units[0]
    hidden2_units = hidden_units[1]

    with tf.name_scope('hidden1') as scope:
        weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units], stddev=1.0/math.sqrt(float(IMAGE_PIXELS))), name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    with tf.name_scope('hidden2') as scope:
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units], stddev=1.0/math.sqrt(float(hidden1_units))), name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    with tf.name_scope('softmax_linear') as scope:
        weights = tf.Variable(tf.truncated_normal([hidden2_units, NUM_CLASSES], stddev=1.0/math.sqrt(float(hidden2_units))), name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        logits = tf.matmul(hidden2, weights) + biases

    return logits


# loss
def loss(logits, labels):
    # batchのサイズはカテゴリー数とする
    batch_size = tf.size(labels)

    # labels labels(batchの各事例に対するlabel tensor)を事例ごとのtensorに分解
    labels = tf.expand_dims(labels, 1)

    # indices batch_sizeのシーケンスをtensorに分解
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)

    # labelsとindicesを結合 batchのindexとlabelのtupleを要素とするtensorを生成
    concated = tf.concat(1, [indices, labels])

    # onehot valuesの生成 packでは[tensorの数、各tensorの次元]のtupleを作成、concatedは[batchのindex, label]のtuple、3rd argsにhotのvalue、4th argsにnon hotのvalue
    onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)
    
    # 交差エントロピー誤差の計算
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, onehot_labels, name='xentropy')

    # 誤差のbatch内平均
    loss = tf.reduce_mean(cross_entroy, name='xentroy_mean')

    return loss

# trian
def train(loss, learning_rate):
    tf.scalar_summary(loss.op.name, loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # trainable=Falseは学習しない変数
    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(loss, global_step=global_step)

    # train operatorを返す
    return train_op

# evaluate
def evaluation(logits, labels_placeholder):
    # 識別モデルの実行には、nn.in_top_kが使える。2rd argsはk
    # top k-位に正解ラベルが入っていた場合にtrueのtensorとなり、
    # batch_sizeの配列で取得できる
    correct = tf.nn.in_top_k(logits, labels, 1)

    # boolをintにして、合計する(ミニバッチの中の正解数になる)
    return tf.reduce_sum(tf.cast(correct, tf.int32))

def test():
    pass
    #inference()

if __name__ == '__main__':
    test() 
