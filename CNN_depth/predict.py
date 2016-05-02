# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import sys
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf

import cv2
from PIL import Image
import model

#import cnn_tiny_settings as settings
#FLAGS = settings.FLAGS


# settings
import settings
FLAGS = settings.FLAGS
BATCH_SIZE = 1

def _variable_with_weight_decay(name, shape, stddev, wd):
    '''
    重み減衰を利用した変数の初期化
    '''
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _variable_on_cpu(name, shape, initializer):
    '''
    CPUメモリに変数をストアする
    '''
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

if __name__ == '__main__':
    test_image = []
    for path in range(1, len(sys.argv)):
        img = cv2.imread(sys.argv[path])
        h, w, c = np.shape(img)
        resize_height = FLAGS.image_height
        resize_width = FLAGS.image_width
        crop_size = FLAGS.crop_size
        img = cv2.resize(img, (resize_height, resize_width))
        img = img.astype(np.float32)

        # Fine Tuningの際は注意すること
        # img -= self.mean
        
        h, w, c = img.shape
        crop_height, crop_width = ((h-crop_size)/2, (w-crop_size)/2)
        img = img[crop_height:crop_height+crop_size, crop_width:crop_width+crop_size, :]
        img = img[None, ...]
        test_image.append(img)
    test_image = np.asarray(test_image)
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    keep_conv = tf.placeholder(tf.float32)
    keep_hidden = tf.placeholder(tf.float32)

    logits, transform_result = model.inference(images, keep_conv, keep_hidden)
    sess = tf.InteractiveSession()

    sess.run(tf.initialize_all_variables())

    pretrain_params = {}
    train_params = {}
    for variable in tf.trainable_variables():
        variable_name = variable.name
        #print("parameter: %s" %(variable_name))
        scope, name = variable_name.split("/")
        target, _ = name.split(":")
        if variable_name.find('spatial_transformer') <  0:
            print("pretrain parameter: %s" %(variable_name))
            pretrain_params[variable_name] = variable
        print("train parameter: %s" %(variable_name))
        train_params[variable_name] = variable
    saver_cnn = tf.train.Saver(pretrain_params)
    saver_transformers = tf.train.Saver(train_params)

    # restore trained model
    print("load pretrain cnn start.")
    ckpt_cnn = tf.train.get_checkpoint_state('pretrain')
    print(ckpt_cnn.model_checkpoint_path)
    if ckpt_cnn and ckpt_cnn.model_checkpoint_path:
        saver_cnn.restore(sess, ckpt_cnn.model_checkpoint_path)
        global_step = ckpt_cnn.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
        print('No checkpoint pretrain file found.')
        quit()
    print("load pretrain cnn done.")
    print("load transformers start.")
    ckpt_transformers = tf.train.get_checkpoint_state('train')
    print(ckpt_transformers.model_checkpoint_path)
    if ckpt_transformers and ckpt_transformers.model_checkpoint_path:
        saver_transformers.restore(sess, ckpt_transformers.model_checkpoint_path)
        global_step_transformers = ckpt_transformers.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
        print('No checkpoint pretrain file found.')
        quit()
    print("load transformers done.")

    # predict
    print('the num of test images: %d' % (len(test_image)))
    for i in range(len(test_image)):
        start_time = time.time()
        softmax = logits.eval(feed_dict={images: test_image[i], keep_conv: 1.0, keep_hidden: 1.0}) [0]
        transform_image = sess.run(transform_result, feed_dict={images: test_image[i], keep_conv: 1.0, keep_hidden: 1.0})
        print(softmax)
        reshape_transform_image = np.reshape(transform_image, (224, 224, 3))
        print(reshape_transform_image.shape)
        disp_image = np.rollaxis(reshape_transform_image, 2)
        print(np.rollaxis(reshape_transform_image, 2).shape)
        print(type(disp_image))
        show_image = Image.fromarray(np.uint8(reshape_transform_image))
        show_image.show()

        pred = np.argmax(softmax)
        duration = time.time() - start_time
        print('category: %i, duration: %f (sec)' % (pred, duration))

