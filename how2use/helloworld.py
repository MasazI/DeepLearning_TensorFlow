#coding: utf-8

import tensorflow as tf

# how2use Session
hello = tf.constant('Hello, TensorFlow!!!')
sess = tf.Session()
print sess.run(hello)
a = tf.constant(10)
print sess.run(a)
b = tf.constant(20)
print sess.run(a+b)


# how2use match_filenames_once
isess = tf.InteractiveSession()
files = tf.train.match_filenames_once('/Users/masai/source/tensorflow/how2use/txt/[0-9].txt')
# 変数の初期化
isess.run(tf.initialize_all_variables())
print isess.run(files)


# how2use string_input_producer
filename_queue = tf.train.string_input_producer(["txt/reader.txt", "txt/reader2.txt"])
reader = tf.TextLineReader() # いろんなリーダがある
key, value = reader.read(filename_queue)
init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # 行数がわかっていれば指定して取得可能、複数ファイルの場合は結合して順番に取得
    # 行数はmax程度の意味合いなので、実際のデータより多くても良い1:
    # ファイルの順番は保証されない
    for i in range(8):
        print sess.run(value)
    coord.request_stop()
    coord.join(threads)
