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

# settings
import settings
FLAGS = settings.FLAGS

TRAIN_DIR = FLAGS.train_dir
MAX_STEPS = FLAGS.max_steps
LOG_DEVICE_PLACEMENT = FLAGS.log_device_placement
TF_RECORDS = FLAGS.train_tfrecords
BATCH_SIZE = FLAGS.batch_size

def train():
    '''
    Train CNN_tiny for a number of steps.
    '''
    with tf.Graph().as_default():
        # globalなstep数
        global_step = tf.Variable(0, trainable=False)

        # 教師データ
        #images, labels = data_inputs.distorted_inputs(TF_RECORDS)
        #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        #trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
        #trX = trX.reshape(-1, 28, 28, 1)
        #teX = teX.reshape(-1, 28, 28, 1)
        mnist_cluttered = np.load('./data/mnist_sequence1_sample_5distortions5x5.npz')
        X_train = mnist_cluttered['X_train']
        y_train = mnist_cluttered['y_train']
        #X_valid = mnist_cluttered['X_valid']
        #y_valid = mnist_cluttered['y_valid']
        #X_test = mnist_cluttered['X_test']
        #y_test = mnist_cluttered['y_test']
        
        # % turn from dense to one hot representation
        Y_train = dense_to_one_hot(y_train, n_classes=10)
        #Y_valid = dense_to_one_hot(y_valid, n_classes=10)
        #Y_test = dense_to_one_hot(y_test, n_classes=10)
        
        # %% Graph representation of our network
        
        # %% Placeholders for 40x40 resolution
        images = tf.placeholder(tf.float32, [None, 1600]) 
        labels = tf.placeholder(tf.float32, [None, 10])


        # create mini_batch
        #datas, targets = trX.(trX, trY, BATCH_SIZE)
        #images = tf.placeholder(tf.float32, [None, 28, 28, 1])
        #labels = tf.placeholder(tf.float32, [None, 10])

        # graphのoutput
        logits = model.inference(images)

        # loss graphのoutputとlabelを利用
        loss = model.loss(logits, labels)

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
            for start, end in zip(range(0, len(trX), BATCH_SIZE), range(BATCH_SIZE, len(trX), BATCH_SIZE)):
                _, loss_value = sess.run([train_op, loss], feed_dict={images: trX[start:end], labels: trY[start:end]})
                
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            # 3回ごと
            if step % 3 == 0:
                # stepごとの事例数 = mini batch size
                #num_examples_per_step = BATCH_SIZE
                num_examples_per_step = 20

                # 1秒ごとの事例数
                examples_per_sec = num_examples_per_step / duration
                
                # バッチごとの時間
                sec_per_batch = float(duration)

                # time, step数, loss, 1秒で実行できた事例数, バッチあたりの時間
                format_str = '$s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                print str(datetime.now()) + ': step' + str(step) + ', loss= '+ str(loss_value) + ' ' + str(examples_per_sec) + ' examples/sec; ' + str(sec_per_batch) + ' sec/batch'

            # 10回ごと
            if step % 10 == 0:
                pass
                #summary_str = sess.run(summary_op)
                # サマリーに書き込む
                #summary_writer.add_summary(summary_str, step)

            if step % 1000 == 0 or (step * 1) == MAX_STEPS:
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
