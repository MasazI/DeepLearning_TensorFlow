#encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# in generary import
import os, sys
import os.path
import time

import tensorflow.python.platform
import numpy
from six.moves import xrange

# tensorflow
import tensorflow as tf

# load data
import logistic_regression

# settings
import cnn_tf_settings as settings
FLAGS = settings.FLAGS

# 学習データplaceholder
import cnn_tf_placeholder as ph

# 学習データのfeeddict
import cnn_tf_feeddict as fd

# モデル
import cnn_tf_model as model

# モデルの精度測定
def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
    # 正解数
    true_count = 0
    
    # epochごとのstep数は全事例数をミニバッチサイズで割った商
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    
    # 事例数をミニバッチサイズで割り切れるサイズに調整
    num_examples = steps_per_epoch * FLAGS.batch_size

    for step in xrange(steps_per_epoch):
        feed_dict = fd.fill_feed_dict(data_set, images_placeholder, labels_placeholder, FLAGS.batch_size, FLAGS.fake_data)
        
        # runにeval_correctを渡す
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples
    print('examples num, correct num, presision: %d\t%d\t%0.04f' % (num_examples, true_count, precision))
    

def run_training():
    data_sets = logistic_regression.load_data_mnist(one_hot=False)

    hiddens = [FLAGS.hidden1, FLAGS.hidden2]

    # Graph実行用のSession
    with tf.Graph().as_default():
        # placeholderの生成
        images_placeholder, labels_placeholder = ph.placeholder_inputs(FLAGS.batch_size, model.IMAGE_PIXELS, model.NUM_CLASSES)

        # logitsの生成
        logits = model.inference(images_placeholder, hiddens)

        # lossの生成
        loss = model.loss(logits, labels_placeholder)

        # trainオペレーションの生成
        train_op = model.train(loss, FLAGS.learning_rate)

        # evaluationオペレーション
        eval_correct = model.evaluation(logits, labels_placeholder)

        # 可視化用サマリ(グラフのビルド時に初期化)
        summary_op = tf.merge_all_summaries()
        
        # modelの保存用オブジェクト
        saver = tf.train.Saver()

        # sessionオブジェクト
        sess = tf.Session()

        # 変数の初期化
        init = tf.initialize_all_variables()
        sess.run(init)

        # 可視化用の変数初期化
        summary_writer = tf.train.SummaryWriter(FLAGS.model_dir, graph_def=sess.graph_def)
        
        # 学習ステップの実行
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            # 現在のステップ用のデータを取得
            feed_dict = fd.fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder, FLAGS.batch_size, FLAGS.fake_data)

            # train_opを実行
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            # 100ステップごとにサマリーを出力
            if step%FLAGS.summary_step == 0:
                # stdout
                print('Step %d: loss = %.2f , %.3f sec' % (step, loss_value, duration))
                
                # 可視化用の変数アップデート
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            # 1000ステップごとにcheckpointを保存(最大ステップに達した際も保存)
            if (step+1)%FLAGS.checkpoint_step == 0 or (step+1) == FLAGS.max_steps:
                saver.save(sess, FLAGS.model_dir, global_step =step)
                
                # Evaluate model
                print('CheckPoint Train Evalation:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.train) 
                print('CheckPoint Valid Evalation:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.validation)
                print('CheckPoint Test Evalation:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.test)

def main(_):
    run_training()
    
def test():
    print("load settings:")
    print(FLAGS.learning_rate)

    print("load dataset:")
    mnist = logistic_regression.load_data_mnist()

    print("place holder")
    images_placeholder, labels_placeholder = ph.placeholder_inputs(FLAGS.batch_size, model.IMAGE_PIXELS, model.NUM_CLASSES)
    
    print("fill feed dict")
    fd.fill_feed_dict(mnist.train, images_placeholder, labels_placeholder, FLAGS.batch_size, FLAGS.fake_data)    

if __name__ == '__main__':
    #test()
    tf.app.run()
