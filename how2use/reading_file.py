#encoding: utf-8

import tensorflow as tf

import numpy as np

from PIL import Image

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def save():
    '''
    tesnsorflow image save
    https://www.tensorflow.org/versions/master/api_docs/python/image.html#images
    '''
    print "main"

    jpgnames_queue = tf.train.string_input_producer(['images/image.jpg', 'images/image2.jpg'])
    pngnames_queue = tf.train.string_input_producer(['images/image.png'])

    reader = tf.WholeFileReader()
    jpg_key, jpg_value = reader.read(jpgnames_queue)
    png_key, png_value = reader.read(pngnames_queue)

    jpg_image = tf.image.decode_jpeg(jpg_value)
    png_image = tf.image.decode_png(png_value)

    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(2):
            jpg_image_eval = jpg_image.eval()
            #png_image_eval = png_image.eval()

            print type(jpg_image_eval)
            print jpg_image_eval.shape
            print jpg_image_eval
            #print png_image_eval.shape

            # disp jpg
            pilimg = Image.fromarray(np.asarray(jpg_image_eval))
            pilimg.show()

            # jpg save as tfrecords
            rows = jpg_image_eval.shape[0]
            cols = jpg_image_eval.shape[1]
            depth = jpg_image_eval.shape[2]
            image_raw = jpg_image_eval.tostring()

            filename = 'sample.tfrecords'
            print 'Writing ' + filename
            writer = tf.python_io.TFRecordWriter(filename)
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'label': _int64_feature(0),
                'image_raw': _bytes_feature(image_raw)
            }))
            writer.write(example.SerializeToString())

        coord.request_stop()
        coord.join(threads)
        sess.close()

def load():
    filename_queue = tf.train.string_input_producer(['sample.tfrecords'])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, dense_keys=['image_raw', 'height', 'width' ,'depth', 'label'], dense_types=[tf.string, tf.int64, tf.int64, tf.int64, tf.int64])

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = features['height']
    width = features['width']
    depth = features['depth']
    label = features['label']

    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(2):
            image_eval = image.eval()
            image_eval_reshape = image_eval.reshape([height.eval(), width.eval(), depth.eval()])
            pilimg = Image.fromarray(np.asarray(image_eval_reshape))
            pilimg.show()
            print image_eval_reshape.shape
            print height.eval()
            print width.eval()
            print depth.eval()
            print label.eval()

        coord.request_stop()
        coord.join(threads)
        sess.close()


def main(argv):
    save()
    load()

if __name__ == '__main__':
    tf.app.run()
