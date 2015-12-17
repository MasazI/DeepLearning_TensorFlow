#encoding: utf-8

from datetime import datetime
import os.path
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile

import numpy as np
from six.moves import xrange
import tensorflow as tf

# model
import caffenet_model as model

# train operation
import caffenet_train_op as op

# inputs
import data_inputs

from data_feed_inputs import ImageInput

# settings
import caffenet_settings as settings
FLAGS = settings.FLAGS

TRAIN_DIR = FLAGS.train_dir
MAX_STEPS = FLAGS.max_steps
LOG_DEVICE_PLACEMENT = FLAGS.log_device_placement

BATCH_SIZE = FLAGS.batch_size

def train():
    '''
    Train Network for a number of steps.
    '''
    with tf.Graph().as_default():
        # globalなstep数
        global_step = tf.Variable(0, trainable=False)

        # 教師データ
        if FLAGS.training_data_type == 0:
            # directory feed inputs
            image_input = ImageInput('caltech_samples_random.txt')
            images  = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.crop_size, FLAGS.crop_size, FLAGS.image_depth))
            labels = tf.placeholder(tf.int32, shape=(FLAGS.batch_size,))
            learning_rate_node = tf.placeholder(tf.float32, shape=[])
        elif FLAGS.training_data_type == 1:
            # tfrecords inputs
            images, labels = data_inputs.distorted_inputs('data/train_caltech.tfrecords')

        # graphのoutput
        logits = model.inference(images)

        # loss graphのoutputとlabelを利用
        loss = model.loss(logits, labels)

        # 学習オペレーション
        train_op = op.train(loss, global_step)

        # バリデーション用 --> evalに移動するべき
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

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
            
            # directory feed inputs
            if FLAGS.training_data_type == 0:
                train_images, train_labels = image_input.batches(FLAGS.batch_size)
                #print train_images
                print train_labels
                _, loss_value = sess.run([train_op, loss], feed_dict={images: train_images, labels:train_labels, learning_rate_node: FLAGS.learning_rate})
            elif FLAGS.training_data_type == 1:
                # tfrecords inputs
                 _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            # 10回ごと
            if step % 10 == 0:
                # stepごとの事例数 = mini batch size
                num_examples_per_step = BATCH_SIZE

                # 1秒ごとの事例数
                examples_per_sec = num_examples_per_step / duration
                
                # バッチごとの時間
                sec_per_batch = float(duration)

                # time, step数, loss, 1秒で実行できた事例数, バッチあたりの時間
                format_str = '$s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                print str(datetime.now()) + ': step' + str(step) + ', loss= '+ str(loss_value) + ' ' + str(examples_per_sec) + ' examples/sec; ' + str(sec_per_batch) + ' sec/batch '

            # 100回ごと
            if step % 100 == 0:
                if FLAGS.training_data_type == 0:
                    summary_str = sess.run(summary_op, feed_dict={images: train_images, labels:train_labels})
                elif FLAGS.training_data_type == 1:
                    summary_str = sess.run(summary_op)
                # サマリーに書き込む
                #print sumary_str
                summary_writer.add_summary(summary_str, step)

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
