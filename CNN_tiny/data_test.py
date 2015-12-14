# encoding: utf-8

import os
import tensorflow.python.platform
import tensorflow as tf
import data


class DataInputTest(tf.test.TestCase):
    '''
    test data inputs from tfrecords.
    '''
    def inputs(self):
        '''
        create inputs mini batch
        '''
        filename_queue = tf.train.string_input_producer(["data/train.tfrecords"], num_epochs=2)
        result = data.read(filename_queue)
        images, sparse_labels = tf.train.shuffle_batch(
            [result.image, result.label], batch_size=2, num_threads=2,
            capacity=3,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1)
        return images, sparse_labels

    def testSimple(self):
        images, labels = self.inputs()
        init_op = tf.initialize_all_variables()
        with self.test_session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                step = 0
                while not coord.should_stop():
                    # mini-batchの各要素を取得
                    image, label = sess.run([images, labels])
                    print label
                    label = tf.expand_dims(label, 1)
                    print label.eval()
                    indices = tf.expand_dims(tf.range(0, 2, 1), 1) 
                    print indices.eval()
                    concated = tf.concat(1, [indices, label])
                    # ここまででmini batchの各indexに対するlabelをもった行列ができる
                    print concated.eval()
                    dense_labels = tf.sparse_to_dense(concated, [2, 5], 1.0, 0.0)
                    print dense_labels.eval()

                    # ここでは1imageにつき1step
                    step += 1
            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (2, step))
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()
            coord.join(threads)
            sess.close()
            

if __name__ == "__main__":
     tf.test.main()
