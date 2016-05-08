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
import data_feed_inputs_nyu
from data_feed_inputs_nyu import ImageInput

# settings
import settings
FLAGS = settings.FLAGS


COARSE_DIR = FLAGS.coarse_dir
REFINE_DIR = FLAGS.refine_dir

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

        # NYU Dataset V2 original size(480 x 640 x 3) -> crop -> (460 x 620 x 3)
        image_input = ImageInput('./data/nyu_depth_v2_labeled.mat')
        print("the number of train data: %d" % (len(image_input.images)))

        images = tf.placeholder(tf.float32, [None, FLAGS.crop_size_height, FLAGS.crop_size_width, FLAGS.image_depth])
        depths = tf.placeholder(tf.float32, [None, 1, 55, 74])
        invalid_depths = tf.placeholder(tf.float32, [None, 1, 55, 74])
        keep_conv = tf.placeholder(tf.float32)
        keep_hidden = tf.placeholder(tf.float32)

        # graphのoutput
        logits, logits_fine = model.inference(images, keep_conv, keep_hidden)

        # loss graphのoutputとlabelを利用
        loss = model.loss(logits, depths, invalid_depths)

        # 学習オペレーション
        train_op = op.train(loss, global_step)

        # サマリー
        summary_op = tf.merge_all_summaries()

        # 初期化オペレーション
        init_op = tf.initialize_all_variables()

        # Session
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT))

        # saver
        #saver = tf.train.Saver(tf.all_variables())

        sess.run(init_op)    

        # coarseとrefineを分けて保存
        coarse_params = {}
        refine_params = {}
        for variable in tf.trainable_variables():
            variable_name = variable.name
            print("parameter: %s" %(variable_name))
            scope, name = variable_name.split("/")
            target, _ = name.split(":")
            if variable_name.find('coarse') >= 0:
                print("coarse parameter: %s" %(variable_name))
                coarse_params[variable_name] = variable
            if variable_name.find('fine') >= 0:
                print("refine parameter: %s" %(variable_name))
                refine_params[variable_name] = variable

        # define saver
        saver_coarse = tf.train.Saver(coarse_params)
        saver_refine = tf.train.Saver(refine_params)

        # fine tune
        if FLAGS.fine_tune:
            # load coarse paramteters
            coarse_ckpt = tf.train.get_checkpoint_state(COARSE_DIR)
            if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
                print("Pretrained coarse Model Loading.")
                saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
                print("Pretrained coarse Model Restored.")
            else:
                print("No Pretrained coarse Model.")

            # load refine parameters
            refine_ckpt = tf.train.get_checkpoint_state(REFINE_DIR)
            if refine_ckpt and refine_ckpt.model_checkpoint_path:
                print("Pretrained refine Model Loading.")
                saver_refine.restore(sess, refine_ckpt.model_checkpoint_path)
                print("Pretrained refine Model Restored.")
            else:
                print("No Pretrained refine Model.")

        # TODO train coarse or refine (change trainable)
        #if not FLAGS.coarse_train:
        #    for val in coarse_params:
        #        print val
        #if not FLAGS.refine_train:
        #    for val in coarse_params:
        #        print val

        # train refine
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # debug
        # サマリーのライターを設定
        #summary_writer = tf.train.SummaryWriter(TRAIN_DIR, graph_def=sess.graph_def)
        #batches = image_input.get_batches(FLAGS.batch_size)a
        #d = np.asarray(batches[0][0])
        #print d.shape
        #a = np.asarray(batches[0][1])
        #print a.shape
        #logits_val, logits_fine_val, loss_value = sess.run([logits, logits_fine, loss], feed_dict={images: batches[0][0], depths: batches[0][1], invalid_depths: batches[0][2], keep_conv: 1.0, keep_hidden: 1.0})
        #print len(logits_val[0])
        #print len(logits_fine_val[0])
        #print loss_value

        # max_stepまで繰り返し学習
        for step in xrange(MAX_STEPS):
            start_time = time.time()
            previous_time = start_time
            index = 0

            batches = image_input.get_batches(FLAGS.batch_size)
            vals = image_input.get_validation()
            for batch in batches:
                train = batch[0]
                depth = batch[1]
                ignore_depth = batch[2]
                _, loss_value = sess.run([train_op, loss], feed_dict={images: train, depths: depth, invalid_depths: ignore_depth, keep_conv: 0.8, keep_hidden: 0.5})
                if index % 10 == 0:
                    end_time = time.time()
                    duration = end_time - previous_time
                    num_examples_per_step = BATCH_SIZE * 10
                    examples_per_sec = num_examples_per_step / duration
                    print("%s: %d[epoch]: %d[iteration]: train loss %f: %d[examples/iteration]: %f[examples/sec]: %f[sec/iteration]" % (datetime.now(), step, index, loss_value, num_examples_per_step, examples_per_sec, duration))
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if index % 50 == 0:
                    output_vec, cost_value = sess.run([logits, loss], feed_dict={images: vals[0], depths: vals[1], invalid_depths: vals[2], keep_conv: 1.0, keep_hidden: 1.0})
                    print("%s: %d[epoch]: %d[iteration]: validation loss: %f" % (datetime.now(), step, index, cost_value))
                    if index % 100 == 0:
                        output_dir = "predicts_%05d_%08d" % (step, index)
                        print("predicts output: %s" % output_dir)
                        data_feed_inputs_nyu.output_predict(output_vec, output_dir)

                previous_time = end_time
                index += 1
                
        #        if index % 100 == 0:
        #            pass
        #            summary_str = sess.run(summary_op, feed_dict={images: train, labels: label, keep_conv: 0.8, keep_hidden: 0.5})
        #            # サマリーに書き込む
        #            summary_writer.add_summary(summary_str, step)
        #    
            if step % 5 == 0 or (step * 1) == MAX_STEPS:
                if FLAGS.coarse_train:
                    coarse_checkpoint_path = COARSE_DIR + '/model.ckpt'
                    saver_coarse.save(sess, coarse_checkpoint_path, global_step=step)
                if FLAGS.refine_train:
                    refine_checkpoint_path = REFINE_DIR + '/model.ckpt'
                    saver_refine.save(sess, refine_checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)
        sess.close()


def main(argv=None):
    if not FLAGS.fine_tune and gfile.Exists(COARSE_DIR):
        #gfile.DeleteRecursively(COARSE_DIR)
        print("Exist directory, please confirm coarse checkpoint.")
        exit(0)

    if not FLAGS.fine_tune and gfile.Exists(REFINE_DIR):
        print("Exist directory, please confirm refine checkpoint.")
        exit(0)

    if not gfile.Exists(COARSE_DIR):
        gfile.MakeDirs(COARSE_DIR)
    if not gfile.Exists(REFINE_DIR):
        gfile.MakeDirs(REFINE_DIR)

    train()


if __name__ == '__main__':
    tf.app.run()