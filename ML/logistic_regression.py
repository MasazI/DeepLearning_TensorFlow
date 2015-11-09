#encoding: utf-8
import input_data
import tensorflow as tf

def load_data_mnist():
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        return mnist

class LogisticRegression():
    def __init__(self): 
        self.sess = tf.InteractiveSession()

        # input data 28x28 for mninst
        self.x = tf.placeholder("float", shape=[None, 784])
        # output data
        self.y_ = tf.placeholder("float", shape=[None, 10])

        self.W = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))    

        self.sess.run(tf.initialize_all_variables())

        # sofmax
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)

        # cross_entropy = negative loss likelihood
        self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y))

    def train(self):
        mnist = load_data_mnist()

        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.cross_entropy)

        for i in range(1000):
            batch = mnist.train.next_batch(50)
            train_step.run(feed_dict={self.x: batch[0], self.y_:batch[1]})
            correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print accuracy.eval(feed_dict={self.x: mnist.test.images, self.y_:mnist.test.labels})

def test():
    classifier = LogisticRegression()
    classifier.train()

if __name__ == '__main__':
    test()
