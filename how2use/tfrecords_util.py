# encoding: utf-8
import tensorflow as tf
import numpy as np
import cv2

import os
import os.path

from PIL import Image

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_file(examples_list_file):
    lines = np.genfromtxt(examples_list_file, delimiter=" ", dtype=[('col1', 'S120'), ('col2', 'i8')])
    examples = []
    labels = []
    for example, label in lines:
        examples.append(example)
        labels.append(label)
    return np.asarray(examples), np.asarray(labels), len(lines)


def extract_image(filename, resize, resize_height, resize_width):
    image = cv2.imread(filename)
    if resize:
        image = cv2.resize(image, (resize_height, resize_width))

    # transform bgr to rgb
    b,g,r = cv2.split(image)       # get b,g,r
    rgb_image = cv2.merge([r,g,b])     # switch it to rgb

    return rgb_image


def transform2tfrecord(examples_list_file, name, output_directory, resize=False, resize_height=256, resize_width=256):
    if not os.path.exists(output_directory) or os.path.isfile(output_directory):
        os.makedirs(output_directory)
    '''
    transform to tfrecords file
    '''
    _examples, _labels, examples_num = load_file(examples_list_file)
    filename = output_directory + "/" + name + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    for i, [example, label] in enumerate(zip(_examples, _labels)):
        print('No.%d' % (i))
        image = extract_image(example, resize, resize_height, resize_width)
        print('shape: %d, %d, %d, label: %d' % (image.shape[0], image.shape[1], image.shape[2], label))
        image_raw = image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'depth': _int64_feature(image.shape[2]),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_raw)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def disp_tfrecords(tfrecord_list_file):
    '''
    show tfrecords image for debug.
    '''
    filename_queue = tf.train.string_input_producer([tfrecord_list_file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        dense_keys=['image_raw', 'height', 'width', 'depth', 'label'],
        dense_types=[tf.string, tf.int64, tf.int64, tf.int64, tf.int64]
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = features['height']
    width = features['width']
    depth = features['depth']
    label = tf.cast(features['label'], tf.int32)
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(4):
            image_eval = image.eval()
            image_eval_reshape = image_eval.reshape([height.eval(), width.eval(), depth.eval()])
            pilimg = Image.fromarray(np.asarray(image_eval_reshape))
            pilimg.show()
            print label.eval()

        coord.request_stop()
        coord.join(threads)
        sess.close()

def read_tfrecord(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        dense_keys=['image_raw', 'height', 'width', 'depth', 'label'],
        dense_types=[tf.string, tf.int64, tf.int64, tf.int64, tf.int64]
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    # image
    tf.reshape(image, [256, 256, 3])
    # normalize
    image = tf.cast(image, tf.float32) * (1. /255) - 0.5
    # label
    label = tf.cast(features['label'], tf.int32)
    return image, label


def test():
    examples_list_file = 'examples.txt'
    #transform2tfrecord(examples_list_file, name='train' , output_directory='./tfrecords', resize=True, resize_height=256, resize_width=256)
    disp_tfrecords('./tfrecords/train.tfrecords')

if __name__ == '__main__':
    test()
