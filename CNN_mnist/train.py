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

def train():
    '''
    Train CNN_tiny for a number of steps.
    '''
    with tf.Graph().as_default():
        # globalなstep数
        global_step = tf.Variable(0, trainable=False)

        # 教師データ
        #images, labels = data_inputs.distorted_inputs(TF_RECORDS)
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
        trX = trX.reshape(-1, 28, 28, 1)
        teX = teX.reshape(-1, 28, 28, 1)

        # create mini_batch
        #datas, targets = trX.(trX, trY, BATCH_SIZE)

        images = tf.placeholder(tf.float32, [None, 28, 28, 1])
        labels = tf.placeholder(tf.float32, [None, 10])

        # graphのoutput
        logits = model.inference(images)

        # loss graphのoutputとlabelを利用
        #loss = model.loss(logits, labels)
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
        train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)
        predict_op = tf.argmax(logits, 1)

        # 学習オペレーション
        #train_op = op.train(loss, global_step)

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
  
        #predict_op = tf.argmax(logits, 1)
 
        # max_stepまで繰り返し学習
        for step in xrange(MAX_STEPS):
            start_time = time.time()

            index = 0
            for start, end in zip(range(0, len(trX), BATCH_SIZE), range(BATCH_SIZE, len(trX), BATCH_SIZE)):
                _, loss_value = sess.run([train_op, loss], feed_dict={images: trX[start:end], labels: trY[start:end]})
                if index % 10 == 0:
                    end_time = time.time()
                    duration = end_time - start_time
                    num_examples_per_step = BATCH_SIZE * 10 * (step+1)
                    examples_per_sec = num_examples_per_step / duration
                    print("%s: %d[epoch]: %d[iteration]: train loss %f: %d[examples/step]: %f[examples/sec]: %f[sec/batch]" % (datetime.now(), step, index, loss_value, num_examples_per_step, examples_per_sec, duration))
                    index += 1
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    test_indices = np.arange(len(teX)) # Get A Test Batch
                    np.random.shuffle(test_indices)
                    test_indices = test_indices[0:5]
                    print "="*20
                    print teY[test_indices]
                    predict, cost_value = sess.run([predict_op, loss], feed_dict={images: teX[test_indices],
                                                                     labels: teY[test_indices]})
                    print predict
                    print("test loss: %f" % (cost_value))
                    print "="*20

                index += 1
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                
                # 1000回ごと
                if index % 1000 == 0:
                    pass
                    #summary_str = sess.run(summary_op)
                    # サマリーに書き込む
                    #summary_writer.add_summary(summary_str, step)
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
