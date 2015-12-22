# encoding: utf-8

import tensorflow.python.platform
import tensorflow as tf

import cnn_tiny_settings as settings
FLAGS = settings.FLAGS

IMAGE_HEIGHT = FLAGS.image_height
IMAGE_WIDTH = FLAGS.image_width
IMAGE_DEPTH = FLAGS.image_depth


def read(filename_queue):
    '''
    read from tfrecords file.
    '''
    class Record(object):
        pass
    result = Record()

    # tfrecords reader
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # parse single example
    features = tf.parse_single_example(
        serialized_example,
        dense_keys=['image_raw', 'height', 'width', 'depth', 'label'],
        dense_types=[tf.string, tf.int64, tf.int64, tf.int64, tf.int64]
    )

    # image
    _image = tf.decode_raw(features['image_raw'], tf.uint8)
    # batch は shape が定義済みであることを求めるため、reshapeする
    shape = [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH]
    result.image = tf.reshape(_image, shape)

    # normalize
    #result.image = tf.cast(image, tf.float32) * (1. /255) - 0.5

    # dense label
    result.label = tf.cast(features['label'], tf.int32)
    return result
