#encoding: utf-8

from datetime import datetime
import os.path
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile

from PIL import Image

import numpy as np
from six.moves import xrange
import tensorflow as tf

# model
import model

# train operation
import train_op as op

# inputs
#import data_inputs
from data_feed_inputs_nyu import ImageInput

# settings
import settings
FLAGS = settings.FLAGS

TRAIN_DIR = FLAGS.train_dir
PRETRAIN_DIR = FLAGS.pretrain_dir
MAX_STEPS = FLAGS.max_steps
LOG_DEVICE_PLACEMENT = FLAGS.log_device_placement
TF_RECORDS = FLAGS.train_tfrecords
BATCH_SIZE = FLAGS.batch_size


def dense_to_one_hot(labels, n_classes=2):
    labels = np.array(labels)
    n_labels = labels.shape[0]
    index_offset = np.arange(n_labels) * n_classes
    labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    print len(labels_one_hot)
    return labels_one_hot


def train():
    '''
    Train
    '''
    with tf.Graph().as_default():
        # globalなstep数
        global_step = tf.Variable(0, trainable=False)

        # NYU Dataset V2 (480 x 640 x 3) -> crop -> (460 x 620 x 3)
        image_input = ImageInput('./data/nyu_depth_v2_labeled.mat')
        print("the number of train data: %d" % (len(image_input.images)))

        images = tf.placeholder(tf.float32, [None, FLAGS.crop_size_height, FLAGS.crop_size_width, FLAGS.image_depth])
        labels = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
        keep_conv = tf.placeholder(tf.float32)
        keep_hidden = tf.placeholder(tf.float32)

        # graphのoutput
        logits = model.inference(images, keep_conv, keep_hidden)

        # loss graphのoutputとlabelを利用
        # loss = model.loss(logits, labels)
        
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

        # 学習オペレーション
        #train_op = op.train(loss, global_step)

        # サマリー
        summary_op = tf.merge_all_summaries()

        # 初期化オペレーション
        init_op = tf.initialize_all_variables()

        # Session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT))

        # saver
        #saver = tf.train.Saver(tf.all_variables())

        sess.run(init_op)    
        # pretrainと全体を分けて保存
        #pretrain_params = {}
        #train_params = {}
        #for variable in tf.trainable_variables():
        #    variable_name = variable.name
        #    #print("parameter: %s" %(variable_name))
        #    scope, name = variable_name.split("/")
        #    target, _ = name.split(":")
        #    if variable_name.find('spatial_transformer') <  0:
        #        print("pretrain parameter: %s" %(variable_name))
        #        pretrain_params[variable_name] = variable
        #    print("train parameter: %s" %(variable_name))
        #    train_params[variable_name] = variable
        #saver_cnn = tf.train.Saver(pretrain_params)
        #saver_transformers = tf.train.Saver(train_params)

        # pretrained_model
        #if FLAGS.fine_tune:
        #    ckpt = tf.train.get_checkpoint_state(PRETRAIN_DIR)
        #    if ckpt and ckpt.model_checkpoint_path:
        #        print("Pretrained Model Loading.")
        #        saver_cnn.restore(sess, ckpt.model_checkpoint_path)
        #        print("Pretrained Model Restored.")
        #    else:
        #        print("No Pretrained Model.")       

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # サマリーのライターを設定
        #summary_writer = tf.train.SummaryWriter(TRAIN_DIR, graph_def=sess.graph_def)

        batches = image_input.get_batches(FLAGS.batch_size)
        logits = sess.run([logits], feed_dict={images: batches[0][0], keep_conv: 1.0, keep_hidden: 1.0})
        print logits 

        # max_stepまで繰り返し学習
        #for step in xrange(MAX_STEPS):
        #    start_time = time.time()
        #    previous_time = start_time
        #    index = 0

        #    batches = image_input.get_batches(FLAGS.batch_size)
        #    for batch in batches:
        #        train = batch[0]
        #        label = batch[1]
        #        _, loss_value = sess.run([train_op, loss], feed_dict={images: train, labels: label, keep_conv: 0.8, keep_hidden: 0.5})
        #        if index % 10 == 0:
        #            end_time = time.time()
        #            duration = end_time - previous_time
        #            num_examples_per_step = BATCH_SIZE * 10
        #            examples_per_sec = num_examples_per_step / duration
        #            print("%s: %d[epoch]: %d[iteration]: train loss %f: %d[examples/iteration]: %f[examples/sec]: %f[sec/iteration]" % (datetime.now(), step, index, loss_value, num_examples_per_step, examples_per_sec, duration))
        #            index += 1
        #            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        #            # test_indices = np.arange(len(teX)) # Get A Test Batch
        #            # np.random.shuffle(test_indices)
        #            # test_indices = test_indices[0:5]
        #            print "="*20
        #            testx = train[0:2]
        #            #print testx
        #            testy = label[0:2]
        #            print np.argmax(testy[0])
        #            print np.argmax(testy[1])
        #            output_vec, predict, cost_value = sess.run([logits, predict_op, loss], feed_dict={images: testx, labels: testy, keep_conv: 1.0, keep_hidden: 1.0})
        #            print predict
        #            print("test loss: %f" % (cost_value))
        #            print "="*20
        #            previous_time = end_time

        #        index += 1
        #        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
        #        
        #        # 100回ごと
        #        if index % 100 == 0:
        #            pass
        #            summary_str = sess.run(summary_op, feed_dict={images: train, labels: label, keep_conv: 0.8, keep_hidden: 0.5})
        #            # サマリーに書き込む
        #            summary_writer.add_summary(summary_str, step)
        #    
        #    if step % 1 == 0 or (step * 1) == MAX_STEPS:
        #        pretrain_checkpoint_path = PRETRAIN_DIR + '/model.ckpt'
        #        train_checkpoint_path = TRAIN_DIR + '/model.ckpt'
        #        saver_cnn.save(sess, pretrain_checkpoint_path, global_step=step)
        #        saver_transformers.save(sess, train_checkpoint_path, global_step=step)
        coord.request_stop()
        coord.join(threads)
        sess.close()


def main(argv=None):
    if gfile.Exists(TRAIN_DIR):
        gfile.DeleteRecursively(TRAIN_DIR)
    gfile.MakeDirs(TRAIN_DIR)
    train()

if __name__ == '__main__':
    tf.app.run()
