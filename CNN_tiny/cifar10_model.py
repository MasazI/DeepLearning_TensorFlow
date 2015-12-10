# encoding: utf-8

# general
import os
import re
import sys

# tensorflow
import tensorflow as tf

# cifar10
import cifar10

# settings
import cifar10_settings as settings
FLAGS = settings.FLAGS

IMAGE_SIZE = FLAGS.image_size
NUM_CLASSES = FLAGS.num_classes
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = FLAGS.num_examples_per_epoch_for_train
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = FLAGS.num_examples_per_epoch_for_eval

MOVING_AVERAGE_DECAY = FLAGS.moving_average_decay
NUM_EPOCHS_PER_DECAY = FLAGS.num_epochs_per_decay
LEARNING_RATE_DECAY_FACTOR = FLAGS.learning_rate_decay_factor
INITIAL_LEARNING_RATE = FLAGS.learning_rate

TOWER_NAME = FLAGS.tower_name




def test():
    pass

if __name__ == '__main__':
    test()
