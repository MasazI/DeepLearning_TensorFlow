#encoding: utf-8

import os
import sys
import cv2
import numpy as np
import tensorflow as tf

# settings
import caffenet_settings as settings
FLAGS = settings.FLAGS

class ImageInput(object):
    def __init__(self, examples_path):
        gt_lines = open(examples_path).readlines()
        gt_pairs = [line.split() for line in gt_lines]
        self.image_paths = [p[0] for p in gt_pairs]
        self.labels = np.array([int(p[1]) for p in gt_pairs])
        #self.model = model
        self.mean = np.array([104., 117., 124.])

    def read_image(self, path):
        img = cv2.imread(path)
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
        return img

    def batches(self, n):
        for i in xrange(0, len(self.image_paths), n):
            images = np.concatenate(map(self.read_image, self.image_paths[i:i+n]), axis=0)
            labels = self.labels[i:i+n]
            return (images, labels)

    def __len__(self):
        return len(self.labels)


def test_imagenet(trained_model_path, examples_path, top_k=3):
    test_data   = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.crop_size, FLAGS.crop_size, FLAGS.image_depth))
    test_labels = tf.placeholder(tf.int32, shape=(FLAGS.batch_size,))
    net         = CaffeNet({'data':test_data})
    probs       = net.get_output()
    top_k_op    = tf.nn.in_top_k(probs, test_labels, top_k)
    imagenet    = ImageInput(examples_path)
    correct     = 0
    count       = 0
    total       = len(imagenet)
    with tf.Session() as sess:
        net.load(trained_model_path, sess)
        for idx, (images, labels) in enumerate(imagenet.batches(FLAGS.batch_size)):
            correct += np.sum(sess.run(top_k_op, feed_dict={test_data:images, test_labels:labels}))
            count += len(images)
            cur_accuracy = float(correct)*100/count
            print('{:>6}/{:<6} {:>6.2f}%'.format(count, total, cur_accuracy))
    print('Top %s Accuracy: %s'%(top_k, float(correct)/total))


def main():
    trained_model_path = 'trained_model/caffenet.npy'
    examples_path = 'examples.txt'
    test_imagenet(trained_model_path, examples_path)


if __name__ == '__main__':
    main()

