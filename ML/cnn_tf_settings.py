# encoding: utf-8

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.01, 'Inirial learning rate.')
flags.DEFINE_integer('max_steps', 10000, 'The number of steps to run trainning.')
flags.DEFINE_integer('hidden1', 128, 'first hidden layer units')
flags.DEFINE_integer('hidden2', 32, 'second hidden layer units')
flags.DEFINE_integer('batch_size', 100, 'Mini Batch size.')
flags.DEFINE_string('train_dir', 'MNIST_data', 'directory of training data.')
flags.DEFINE_string('model_dir', 'models', 'directory of model data.')
flags.DEFINE_integer('summary_step', 100, 'The number of steps to print sammary.')
flags.DEFINE_integer('checkpoint_step', 1000, 'The number of steps to save checkpoint.')
flags.DEFINE_boolean('fake_data', False, 'true: use fake data for unit testing.')
