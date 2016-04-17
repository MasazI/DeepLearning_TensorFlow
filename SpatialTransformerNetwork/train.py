#encoding: utf-8

from datetime import datetime
import os.path
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile

from PIL import Image

import numpy as np
from six.moves import xrange
import tensorflow as tf

# model
import model

# train operation
import train_op as op

# inputs
#import data_inputs
import input_data

# settings
import settings
FLAGS = settings.FLAGS

TRAIN_DIR = FLAGS.train_dir
MAX_STEPS = FLAGS.max_steps
LOG_DEVICE_PLACEMENT = FLAGS.log_device_placement
TF_RECORDS = FLAGS.train_tfrecords
BATCH_SIZE = FLAGS.batch_size


def dense_to_one_hot(labels, n_classes=2):
    labels = np.array(labels)
    n_labels = labels.shape[0]
    index_offset = np.arange(n_labels) * n_classes
    labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    print len(labels_one_hot)
    return labels_one_hot


def train():
    '''
    Train CNN_tiny for a number of steps.
    '''
    with tf.Graph().as_default():
        # globalなstep数
        global_step = tf.Variable(0, trainable=False)

        # 教師データ
        #images, labels = data_inputs.distorted_inputs(TF_RECORDS)
        # 教師データ
        mnist = np.load('./data/mnist_sequence1_sample_5distortions5x5.npz')
        trX = mnist['X_train']
        trY = mnist['y_train']
        # X_valid = mnist_cluttered['X_valid']
        # y_valid = mnist_cluttered['y_valid']
        teX = mnist['X_test']
        teY = mnist['y_test']
        trX = trX.reshape(-1, 40, 40, 1)
        teX = teX.reshape(-1, 40, 40, 1)
        
        # % turn from dense to one hot representation
        trY = dense_to_one_hot(trY, n_classes=10)
        trY = trY.reshape(-1, 10)
        # Y_valid = dense_to_one_hot(y_valid, n_classes=10)
        teY = dense_to_one_hot(teY, n_classes=10)
        teY = teY.reshape(-1, 10)

        print("the number of train data: %d" % (len(trX)))

        # create mini_batch
        #datas, targets = trX.(trX, trY, BATCH_SIZE)

        images = tf.placeholder(tf.float32, [None, 40, 40, 1])
        labels = tf.placeholder(tf.float32, [None, 10])
        keep_conv = tf.placeholder(tf.float32)
        keep_hidden = tf.placeholder(tf.float32)

        # graphのoutput
        logits = model.inference(images, keep_conv, keep_hidden)

        # loss graphのoutputとlabelを利用
        #loss = model.loss(logits, labels)
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
        predict_op = tf.argmax(logits, 1)

        # 学習オペレーション
        train_op = op.train(loss, global_step)

        # saver
        saver = tf.train.Saver(tf.all_variables())

        # サマリー
        summary_op = tf.merge_all_summaries()

        # 初期化オペレーション
        init_op = tf.initialize_all_variables()

        # Session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT))
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # サマリーのライターを設定
        summary_writer = tf.train.SummaryWriter(TRAIN_DIR, graph_def=sess.graph_def)
 
        # max_stepまで繰り返し学習
        for step in xrange(MAX_STEPS):
            start_time = time.time()
            previous_time = start_time
            index = 0
            for start, end in zip(range(0, len(trX), BATCH_SIZE), range(BATCH_SIZE, len(trX), BATCH_SIZE)):
                _, loss_value = sess.run([train_op, loss], feed_dict={images: trX[start:end], labels: trY[start:end], keep_conv: 0.8, keep_hidden: 0.5})
                if index % 10 == 0:
                    end_time = time.time()
                    duration = end_time - previous_time
                    num_examples_per_step = BATCH_SIZE * 10 * (step+1)
                    examples_per_sec = num_examples_per_step / duration
                    print("%s: %d[epoch]: %d[iteration]: train loss %f: %d[examples/step]: %f[examples/sec]: %f[sec/iteration]" % (datetime.now(), step, index, loss_value, num_examples_per_step, examples_per_sec, duration))
                    index += 1
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    test_indices = np.arange(len(teX)) # Get A Test Batch
                    np.random.shuffle(test_indices)
                    test_indices = test_indices[0:5]
                    print "="*20
                    print teY[test_indices]
                    predict, cost_value = sess.run([predict_op, loss], feed_dict={images: teX[test_indices], labels: teY[test_indices], keep_conv: 1.0, keep_hidden: 1.0})
                    print predict
                    print("test loss: %f" % (cost_value))
                    print "="*20
                    previous_time = end_time

                index += 1
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                
                # 1000回ごと
                if index % 100 == 0:
                    pass
                    summary_str = sess.run(summary_op, feed_dict={images: trX[start:end], labels: trY[start:end], keep_conv: 0.8, keep_hidden: 0.5})
                    # サマリーに書き込む
                    summary_writer.add_summary(summary_str, step)
            if step % 1 == 0 or (step * 1) == MAX_STEPS:
                checkpoint_path = TRAIN_DIR + '/model.ckpt'
                saver.save(sess, checkpoint_path, global_step=step)
        coord.request_stop()
        coord.join(threads)
        sess.close()


def main(argv=None):
    if gfile.Exists(TRAIN_DIR):
        gfile.DeleteRecursively(TRAIN_DIR)
    gfile.MakeDirs(TRAIN_DIR)
    train()

if __name__ == '__main__':
    tf.app.run()
