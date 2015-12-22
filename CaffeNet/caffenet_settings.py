# encoding: utf-8

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# train settings
flags.DEFINE_integer('batch_size', 40, 'the number of images in a batch.')
flags.DEFINE_integer('training_data_type', 1, '0: directly feed, 1: tfrecords')
#flags.DEFINE_string('train_tfrecords', 'data/train_caltech_random.tfrecords', 'path to tfrecords file for train.')
flags.DEFINE_string('train_tfrecords', 'data/train_ex_norm.tfrecords', 'path to tfrecords file for train.')
flags.DEFINE_integer('image_height', 256, 'image height.')
flags.DEFINE_integer('image_width', 256, 'image width.')
flags.DEFINE_integer('image_depth', 3, 'image depth.')
flags.DEFINE_integer('crop_size', 227, 'crop size of image.')
flags.DEFINE_float('learning_rate', 1e-2, 'initial learning rate.')
flags.DEFINE_float('learning_rate_decay_factor', 0.1, 'learning rate decay factor.')
flags.DEFINE_float('num_epochs_per_decay', 350.0, 'epochs after which learning rate decays.')
flags.DEFINE_float('moving_average_decay', 0.9999, 'decay to use for the moving averate.')
flags.DEFINE_integer('num_examples_per_epoch_for_train', 400, 'the number of examples per epoch train.')
flags.DEFINE_integer('num_examples_per_epoch_for_eval', 400, 'the number of examples per eposh eval.')
flags.DEFINE_string('tower_name', 'tower', 'multiple GPU prefix.')
#flags.DEFINE_integer('num_classes', 10, 'the number of classes.')
flags.DEFINE_integer('num_classes', 5, 'the number of classes.')
flags.DEFINE_integer('num_threads', 8, 'the number of threads.')
flags.DEFINE_boolean('fine_tuning', False, 'fine tuning.')
flags.DEFINE_string('trained_model', 'trained_model/caffenet.npy' , 'trained model to use fine tuning.')

# output logs settings
flags.DEFINE_string('train_dir', 'train', 'directory where to write even logs and checkpoint')
flags.DEFINE_integer('max_steps', 100000, 'the number of batches to run.')
flags.DEFINE_boolean('log_device_placement', False, 'where to log device placement.')

# evaluate settings
flags.DEFINE_string('eval_dir', 'eval', 'directory where to write event logs.')
flags.DEFINE_string('eval_tfrecords', 'data/train_ex_norm.tfrecords', 'path to tfrecords file for eval')
flags.DEFINE_string('checkpoint_dir', 'train', 'directory where to read model checkpoints.')
flags.DEFINE_integer('eval_interval_secs', 60*3, 'How to often to run the eval.'),
flags.DEFINE_integer('num_examples', 100, 'the number of examples to run.')
flags.DEFINE_boolean('run_once', False, 'whether to run eval only once.')
