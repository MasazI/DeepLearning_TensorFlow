# encoding: utf-8

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 128, 'the number of images in a batch.')
flags.DEFINE_string('data_dir', 'cifar10_data', 'path to the CIFAR-10 directory.')
flags.DEFINE_integer('image_size', 24, 'image size.')
flags.DEFINE_float('learning_rate', 0.1, 'initial learning rate.')
flags.DEFINE_float('learning_rate_decay_factor', 0.1, 'learning rate decay factor.')
flags.DEFINE_float('num_epochs_per_decay', 350.0, 'epochs after which learning rate decays.')
flags.DEFINE_float('moving_average_decay', 0.9999, 'decay to use for the moving averate.')
flags.DEFINE_integer('num_examples_per_epoch_for_train', 50000, 'the number of examples per epoch train.')
flags.DEFINE_integer('num_examples_per_epoch_for_eval', 1000, 'the number of examples per eposh eval.')
flags.DEFINE_string('tower_name', 'tower', 'multiple GPU prefix')
flags.DEFINE_integer('num_classes', 10, 'the number of classes')
