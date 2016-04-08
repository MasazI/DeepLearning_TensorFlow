# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import sys
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf

import load

import model_mlp

NUM_CLASSES = 1
IMAGE_SIZE = 8
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

# 学習時のbatch size
BATCH_SIZE = 1

if __name__ == '__main__':
    filename_queue = tf.train.string_input_producer(["data/airquality.csv"])
    feature, data = load.mini_batch(filename_queue, 1)
    data_placeholder = tf.placeholder("float", shape=(None, 5))
    
    logits = model_mlp.inference(data_placeholder)
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    # restore trained model
    ckpt = tf.train.get_checkpoint_state('train')
    print(ckpt.model_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        print("Load checkpint.")
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
        print('No checkpoint file found.')
        quit()

    # create threads
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # predict
    for i in xrange(10):
        start_time = time.time()
        x, y = sess.run([feature, data])
        predict_value = logits.eval(feed_dict={data_placeholder: x})
        duration = time.time() - start_time
        print('No.%d: feature: %s, truth: %f, predict: %f, duration: %f (sec)' % (i, x, y, predict_value, duration))

    coord.request_stop()
    coord.join(threads)

