# encoding: utf-8

import tensorflow.python.platform
import tensorflow as tf

import settings
FLAGS = settings.FLAGS

import numpy as np

def read(filename_queue):
    '''
    read from tfrecords file.
    '''
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    
    record_defaults = [[.0], [.0], [.0], [.0], [.0], [.0]]
    target, col1, col2, col3, col4, col5  = tf.decode_csv(value, record_defaults=record_defaults)
    features = tf.pack([col1, col2, col3, col4, col5])
    targets = tf.pack([target])
    
    return features, targets

def mini_batch(filename_queue, mini_batch_size):
    feature, target = read(filename_queue)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * mini_batch_size
    features, targets = tf.train.shuffle_batch([feature, target], batch_size=mini_batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

    return features, targets


if __name__ == '__main__':
    filename_queue = tf.train.string_input_producer(["data/airquality.csv"])
    features, targets = mini_batch(filename_queue, 1)


    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(100):
            x, y = sess.run([features, targets])
            print x, y

        coord.request_stop()
        coord.join(threads)
