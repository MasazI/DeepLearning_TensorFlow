#encoding: utf-8

from datetime import datetime
import os.path
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile

#from PIL import Image

import numpy as np
from six.moves import xrange
import tensorflow as tf

# model
import model_mlp as model

# train operation
import train_op as op

# inputs
import load

# settings
import settings
FLAGS = settings.FLAGS

TRAIN_DIR = FLAGS.train_dir
MAX_STEPS = FLAGS.max_steps
LOG_DEVICE_PLACEMENT = FLAGS.log_device_placement
TF_RECORDS = FLAGS.train_tfrecords
BATCH_SIZE = FLAGS.batch_size

from datetime import datetime as dt
tdatetime = dt.now()
train_start_time = tdatetime.strftime('%Y%m%d%H%M%S')

def train():
    '''
    Train CNN_tiny for a number of steps.
    '''
    with tf.Graph().as_default():
        # globalなstep数
        global_step = tf.Variable(0, trainable=False)

        # 教師データ
        filename_queue = tf.train.string_input_producer(["data/airquality.csv"])
        datas, targets = load.mini_batch(filename_queue, BATCH_SIZE)

        # placeholder
        x = tf.placeholder(tf.float32, shape=[None, 5])
        y = tf.placeholder(tf.float32, shape=[None, 1])

        # graphのoutput
        logits_maru = model.inference(x, 'maru')
        logits_batsu = model.inference(x, 'batsu')

        debug_value_maru = model.debug(logits_maru)
        debug_value_batsu = model.debug(logits_batsu)

        # loss graphのoutputとlabelを利用
        loss_maru = model.loss(logits_maru, y, 'maru')
        loss_batsu = model.loss(logits_batsu, y, 'batsu')

        # 学習オペレーション
        train_op_maru = op.train(loss_maru, global_step)
        train_op_batsu = op.train(loss_batsu, global_step)

        # saver
        saver = tf.train.Saver(tf.all_variables())

        # サマリー
        summary_op = tf.merge_all_summaries()

        # 初期化オペレーション
        init_op = tf.initialize_all_variables()

        # Session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT))
        sess.run(init_op)

        print("settion start.")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # サマリーのライターを設定
        summary_writer = tf.train.SummaryWriter(TRAIN_DIR, graph_def=sess.graph_def)

        # model名
        model_name = '/model%s.ckpt' % (tdatetime.strftime('%Y%m%d%H%M%S'))
   
        # max_stepまで繰り返し学習
        for step in xrange(MAX_STEPS):
            start_time = time.time()
            a, b = sess.run([datas, targets])
            _a, loss_value_maru, predict_value_maru = sess.run([train_op_maru, loss_maru, debug_value_maru], feed_dict={x: a, y: b})
            _b, loss_value_batsu, predict_value_batsu = sess.run([train_op_batsu, loss_batsu, debug_value_batsu], feed_dict={x: a, y: b})

            duration = time.time() - start_time

            assert not np.isnan(loss_value_maru), 'maru Model diverged with loss = NaN'
            assert not np.isnan(loss_value_batsu), 'batsu Model diverged with loss = NaN'

            # 100回ごと
            if step % 100 == 0:
                # stepごとの事例数 = mini batch size
                num_examples_per_step = BATCH_SIZE

                # 1秒ごとの事例数
                examples_per_sec = num_examples_per_step / duration
                
                # バッチごとの時間
                sec_per_batch = float(duration)

                # time, step数, loss, 1秒で実行できた事例数, バッチあたりの時間
                format_str = '$s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                print 'maru' + str(datetime.now()) + ': step' + str(step) + ', loss= '+ str(loss_value_maru) + ' ' + str(examples_per_sec) + ' examples/sec; ' + str(sec_per_batch) + ' sec/batch'
                print 'batsu' + str(datetime.now()) + ': step' + str(step) + ', loss= '+ str(loss_value_batsu) + ' ' + str(examples_per_sec) + ' examples/sec; ' + str(sec_per_batch) + ' sec/batch'


                print "x", a
                print "ground truth:", b
                print "predict maru: ", predict_value_maru
                print "predict batsu: ", predict_value_batsu

            # 100回ごと
            if step % 100 == 0:
                pass
                #summary_str = sess.run(summary_op)
                # サマリーに書き込む
                #summary_writer.add_summary(summary_str, step)

            if step % 1000 == 0 or (step * 1) == MAX_STEPS:
                checkpoint_path = TRAIN_DIR + model_name
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
