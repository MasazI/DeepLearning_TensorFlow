# encoding: utf-8

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 1, 'the number of images in a batch.')
flags.DEFINE_string('train_tfrecords', 'data/train.tfrecords', 'path to tfrecords file for train.')
flags.DEFINE_string('test_tfrecords', '', 'path to tfrecords file for test.')
flags.DEFINE_integer('image_height', 256, 'image height.')
flags.DEFINE_integer('image_width', 256, 'image width.')
flags.DEFINE_integer('image_depth', 3, 'image depth.')
flags.DEFINE_integer('crop_size', 224, 'crop size of image.')
flags.DEFINE_float('learning_rate', 0.1, 'initial learning rate.')
flags.DEFINE_float('learning_rate_decay_factor', 0.1, 'learning rate decay factor.')
flags.DEFINE_float('num_epochs_per_decay', 350.0, 'epochs after which learning rate decays.')
flags.DEFINE_float('moving_average_decay', 0.9999, 'decay to use for the moving averate.')
flags.DEFINE_integer('num_examples_per_epoch_for_train', 1, 'the number of examples per epoch train.')
flags.DEFINE_integer('num_examples_per_epoch_for_eval', 1, 'the number of examples per eposh eval.')
flags.DEFINE_string('tower_name', 'tower', 'multiple GPU prefix.')
flags.DEFINE_integer('num_classes', 4, 'the number of classes.')
flags.DEFINE_integer('num_threads', 1, 'the number of threads.')
