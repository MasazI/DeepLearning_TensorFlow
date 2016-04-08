# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf

import model_alex2010 as model
import data_inputs

import settings
FLAGS = settings.FLAGS

def eval_once(saver, summary_writer, top_k_op, summary_op):
    '''
    run eval once.
    '''
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found.')
            return

        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
            # バッチごとの事例数
            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))

            true_count = 0
            total_sample_count = num_iter * FLAGS.batch_size
            print('the number of total sample count: %d' % (total_sample_count))
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f, %d/%d' % (datetime.now(), precision, true_count, total_sample_count))
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    with tf.Graph().as_default():
        # testデータのロード
        images, labels = data_inputs.inputs('data/test.tfrecords')
        logits = model.inference(images)

        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
        variables_to_restore = {}
        for v in tf.trainable_variables():
            if v in tf.trainable_variables():
                restore_name = variable_averages.average_name(v)
            else:
                restore_name = v.op.name
            variables_to_restore[restore_name] = v
        saver = tf.train.Saver(variables_to_restore)
        summary_op = tf.merge_all_summaries()

        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, graph_def=graph_def)

        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.run()
