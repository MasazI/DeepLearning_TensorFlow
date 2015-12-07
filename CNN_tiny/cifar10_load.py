# encoding: utf-8

import tensorflow.python.platform
import tensorflow as tf

class CIFAR10Record(object):
    def __init__(self):
        pass

def read_cifar10(filename_queue):
    result = CIFAR10Record()
    label_bytes = 1 # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)

    result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]), [result.depth, result.height, result.width])

    result.unit8image = tf.transpose(depth_major, [1,2,0])

    return result


if __name__ == '__main__':
    import os
    import cifar10_settings as settings
    from tensorflow.python.platform import gfile
    FLAGS = settings.FLAGS
    filenames = [os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin', 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
    for f in filenames:
        if not gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)
    
    result = read_cifar10(filename_queue)
    print result.height
