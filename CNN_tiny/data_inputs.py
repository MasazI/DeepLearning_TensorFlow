#encoding: utf-8

# general
import os
import re
import sys

# tensorflow
import tensorflow as tf

# data
import data

# settings
import cnn_tiny_settings as settings
FLAGS = settings.FLAGS

CROP_SIZE = FLAGS.crop_size
BATCH_SIZE = FLAGS.batch_size
NUM_THREADS = FLAGS.num_threads

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = FLAGS.num_examples_per_epoch_for_train

def _generate_image_and_label_batch(image, label, min_queue_examples):
    '''
    imageとlabelのmini batchを生成
    '''
    num_preprocess_threads = NUM_THREADS
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=BATCH_SIZE,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * BATCH_SIZE,
        min_after_dequeue=min_queue_examples
    )

    # Display the training images in the visualizer
    tf.image_summary('images', images)
    return images, tf.reshape(label_batch, [BATCH_SIZE])
    

def distorted_inputs(tfrecords_file):
    '''
    create inputs with real time augumentation.
    '''
    print tfrecords_file
    filename_queue = tf.train.string_input_producer([tfrecords_file]) # ここで指定したepoch数はtrainableになるので注意
    read_input = data.read(filename_queue)
    reshaped_image = tf.cast(read_input.image, tf.float32)

    height = CROP_SIZE
    width = CROP_SIZE

    # crop
    distorted_image = tf.image.random_crop(reshaped_image, [height, width]) 

    # flip
    distorted_image = tf.image.random_flip_left_right(distorted_image)
     
    # you can add random brightness contrast
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    
    # whitening
    float_image = tf.image.per_image_whitening(distorted_image)

    min_fraction_of_examples_in_queue = 0.4
    #min_fraction_of_examples_in_queue = 1
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print ('filling queue with %d train images before starting to train.  This will take a few minutes.' % min_queue_examples)

    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples)


def inputs(tfrecords_file):
    '''
    create inputs
    '''
    print tfrecords_file
    filename_queue = tf.train.string_input_producer([tfrecords_file]) # ここで指定したepoch数はtrainableになるので注意
    read_input = data.read(filename_queue)
    reshaped_image = tf.cast(read_input.image, tf.float32)

    height = CROP_SIZE
    width = CROP_SIZE

    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, width, height)
    float_image = tf.image.per_image_whitening(resized_image)

    min_fraction_of_examples_in_queue = 0.4
    #min_fraction_of_examples_in_queue = 1
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print ('filling queue with %d train images before starting to train.  This will take a few minutes.' % min_queue_examples)

    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples)

