#coding: utf-8

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!!!')
sess = tf.Session()

print sess.run(hello)

a = tf.constant(10)
print sess.run(a)

b = tf.constant(20)

print sess.run(a+b)

isess = tf.InteractiveSession()
files = tf.train.match_filenames_once('/Users/masai/source/tensorflow/how2use/[0-9].txt')

# 変数の初期化
isess.run(tf.initialize_all_variables())
print isess.run(files)

