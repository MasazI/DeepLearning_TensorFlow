# encoding: utf-8

import tensorflow as tf

import cnn_tiny_settings as settings
FLAGS = settings.FLAGS

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = FLAGS.num_examples_per_epoch_for_train
NUM_EPOCHS_PER_DECAY = FLAGS.num_epochs_per_decay
INITIAL_LEARNING_RATE = FLAGS.learning_rate
LEARNING_RATE_DECAY_FACTOR = FLAGS.learning_rate_decay_factor
MOVING_AVERAGE_DECAY = FLAGS.moving_average_decay

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
        print(var.op.name)
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
