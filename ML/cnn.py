#encoding:utf-8
import tensorflow as tf
import logistic_regression

def weight_variable(shape):
    # 初期値を生成したweight
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # 初期値を指定したbias
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

class CNN():
    def __init__(self):
        self.sess = tf.InteractiveSession()

        # input data 28x28 for mninst
        self.x = tf.placeholder("float", shape=[None, 784])
        # output data
        self.y_ = tf.placeholder("float", shape=[None, 10])

        self.W_conv1 = weight_variable([5, 5, 1, 32])
        self.b_conv1 = bias_variable([32])

        x_image = tf.reshape(self.x, [-1, 28, 28, 1])

        self.h_conv1 = tf.nn.relu(conv2d(x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)

        self.W_conv2 = weight_variable([5, 5, 32, 64])
        self.b_conv2 = bias_variable([64])

        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = max_pool_2x2(self.h_conv2)

        self.W_fc1 = weight_variable([7*7*64, 1024])
        self.b_fc1 = bias_variable([1024])

        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        self.keep_prob = tf.placeholder("float")

        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        self.W_fc2 = weight_variable([1024, 10])
        self.b_fc2 = bias_variable([10])

        self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)

    def train(self):
        mnist = logistic_regression.load_data_mnist()

        # negative loss likelihood
        cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y_conv))
        # 最適化関数にコストを渡す
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        # 正しいかどうか
        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # 全変数の初期化
        self.sess.run(tf.initialize_all_variables())

        for i in range(1000):
            batch = mnist.train.next_batch(50)
            if i%100 == 0:
                # 100回ごとにテスト
                train_accuracy = accuracy.eval(feed_dict={self.x: batch[0], self.y_:batch[1], self.keep_prob: 1.0})
                print "setp %d, training accuracy %g" % (i, train_accuracy)
            train_step.run(feed_dict={self.x:batch[0], self.y_:batch[1], self.keep_prob: 1.0})
        print "test accuracy %g" % accuracy.eval(feed_dict={self.x: mnist.test.images, self.y_: mnist.test.labels, self.keep_prob: 1.0})

    
def test():
    cnn = CNN()
    cnn.train()    

if __name__ == '__main__':
    test()
