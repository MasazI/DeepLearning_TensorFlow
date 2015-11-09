#coding: utf-8

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!!!')
sess = tf.Session()

print sess.run(hello)

a = tf.constant(10)
print sess.run(a)

b = tf.constant(20)

print sess.run(a+b)
