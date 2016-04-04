# encoding: utf-8

import tensorflow.python.platform
import tensorflow as tf

import settings
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
    if tf.__version__[2] == '7':
        datas = tf.parse_single_example(
            serialized_example,
            features = {
                "image_raw": tf.FixedLenFeature([], dtype=tf.string),
                "height": tf.FixedLenFeature([], dtype=tf.int64),
                "width": tf.FixedLenFeature([], dtype=tf.int64),
                "depth": tf.FixedLenFeature([], dtype=tf.int64),
                "label": tf.FixedLenFeature([], dtype=tf.int64),
            }
        )
    else:
        datas = tf.parse_single_example(
            serialized_example,
            #dense_keys=['image_raw', 'height', 'width', 'depth', 'label'],
            dense_keys=['image_raw', 'height', 'width', 'depth', 'label'],
            dense_types=[tf.string, tf.int64, tf.int64, tf.int64, tf.int64]
        )

    # image
    _image = tf.decode_raw(datas['image_raw'], tf.uint8)
    # batch は shape が定義済みであることを求めるため、reshapeする
    shape = [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH]
    result.image = tf.reshape(_image, shape)

    # normalize
    #result.image = tf.cast(image, tf.float32) * (1. /255) - 0.5

    # dense label
    result.label = tf.cast(datas['label'], tf.int32)
    return result
