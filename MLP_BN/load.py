# encoding: utf-8

import tensorflow.python.platform
import tensorflow as tf

import settings
FLAGS = settings.FLAGS


def read(filename_queue):
    '''
    read from tfrecords file.
    '''
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    
    record_defaults = [[.0], [.0], [.0], [.0], [.0], [.0]]
    target, col1, col2, col3, col4, col5  = tf.decode_csv(value, record_defaults=record_defaults)
    features = tf.pack([col1, col2, col3, col4, col5])

    return features, target 

if __name__ == '__main__':
    filename_queue = tf.train.string_input_producer(["data/airquality.csv"])
    features, target = read(filename_queue)


    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(100):
            x, y = sess.run([features, target])
            print tf.rank(x)
            print tf.rank(y)
            print x, y

        coord.request_stop()
        coord.join(threads)
