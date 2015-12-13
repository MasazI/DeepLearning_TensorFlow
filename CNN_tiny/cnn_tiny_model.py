# encoding: utf-8

# general
import os
import re
import sys

# tensorflow
import tensorflow as tf

# data
import data

# settings
import cnn_tiny_settings as settings
FLAGS = settings.FLAGS

NUM_CLASSES = FLAGS.num_classes
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = FLAGS.num_examples_per_epoch_for_train
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = FLAGS.num_examples_per_epoch_for_eval

MOVING_AVERAGE_DECAY = FLAGS.moving_average_decay
NUM_EPOCHS_PER_DECAY = FLAGS.num_epochs_per_decay
LEARNING_RATE_DECAY_FACTOR = FLAGS.learning_rate_decay_factor
INITIAL_LEARNING_RATE = FLAGS.learning_rate
CROP_SIZE = FLAGS.crop_size

BATCH_SIZE = FLAGS.batch_size
NUM_THREADS = FLAGS.num_threads

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


def _generate_image_and_label_batch(image, label, min_queue_examples):
    '''
    imageとlabelのmini batchを生成
    '''
    num_preprocess_threads = NUM_THREADS
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=BATCH_SIZE,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * BATCH_SIZE,
        min_after_dequeue=min_queue_examples
    )

    # Display the training images in the visualizer
    tf.image_summary('images', images)
    return images, tf.reshape(label_batch, [BATCH_SIZE])
    

def distorted_inputs(tfrecords_file):
    '''
    create inputs with real time augumentation.
    '''
    print tfrecords_file
    filename_queue = tf.train.string_input_producer([tfrecords_file]) # ここで指定したepoch数はtrainableになる
    read_input = data.read(filename_queue)
    reshaped_image = tf.cast(read_input.image, tf.float32)

    height = CROP_SIZE
    width = CROP_SIZE

    # crop
    distorted_image = tf.image.random_crop(reshaped_image, [height, width]) 

    # flip
    distorted_image = tf.image.random_flip_left_right(distorted_image)
     
    # you can add random brightness contrast
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    
    # whitening
    float_image = tf.image.per_image_whitening(distorted_image)

    #min_fraction_of_examples_in_queue = 0.4
    min_fraction_of_examples_in_queue = 1
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print ('filling queue with %d train images before starting to train.  This will take a few minutes.' % min_queue_examples)

    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples)


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
        conv1 = tf.nn.relu(bias, name=scope.name)
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
            stddev=0.04,
            wd=0.004
        )
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu_layer(reshape, weights, biases, name=scope.name)
        _activation_summary(local3)

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
        _activation_summary(local4)

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
        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 1])
    indices = tf.reshape(tf.range(0, FLAGS.batch_size), [FLAGS.batch_size, 1])
    concated = tf.concat(1, [indices, sparse_labels])
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


def train(total_loss, global_step):
    # epochあたりのmini batch数
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    # 重み減衰のステップ 減衰あたりのmini batch数
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # 学習率をステップ数に応じて減衰する
    lr = tf.train.exponential_decay(
        INITIAL_LEARNING_RATE,
        global_step,
        decay_steps,
        LEARNING_RATE_DECAY_FACTOR,
        staircase=True)
    tf.scalar_summary('learning_rate', lr)
    
    # lossの移動平均とサマリーをひも付け
    loss_averages_op = _add_loss_summaries(total_loss)
    
    # 勾配の計算
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    
    # 勾配を適用
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    
    # Add histograms for trainable variables.
    # 学習パラメータのヒストグラムに加える
    for var in tf.trainable_variables():
        # tag, values
        print var.op.name
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    # 勾配のヒストグラムに加える
    for grad, var in grads:
        if grad:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    # 全ての学習パラメータの移動平均をトラックする
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    
    # 学習オペレーションを返す
    return train_op


def test():
    distorted_inputs('data/train.tfrecords')

if __name__ == '__main__':
    test()
