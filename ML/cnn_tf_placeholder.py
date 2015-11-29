#encoding:utf-8

import tensorflow as tf

def placeholder_inputs(batch_size, image_size, num_classes):
    '''
    definition of placeholder for inputs
    '''
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, image_size))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder
