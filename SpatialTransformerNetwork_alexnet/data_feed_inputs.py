#encoding: utf-8

import cv2
import numpy as np

import random

import time
# settings
import settings
FLAGS = settings.FLAGS


class ImageInput(object):
    def __init__(self, examples_path):
        '''
        初期化
        arguments:
            examples_path: 画像リストのパス
        '''
        gt_tmps = open(examples_path).readlines()
        gt_lines = [x.rstrip("\n") for x in gt_tmps]
        gt_pairs = [line.split(',') for line in gt_lines]
        self.image_paths = [p[0] for p in gt_pairs]
        self.labels = np.array([p[1] for p in gt_pairs])
        set_labels = set(self.labels)
        self.list_labels = [label for label in set_labels]
        self.num_classes = len(self.list_labels)
        self.batches = []
        print("the number of training sets: %d" % (len(self.image_paths)))
        print("the number of traing labels: %d" % (len(self.labels)))
        print("the number of classes: %d" % (len(self.list_labels)))
        # 平均
        # self.mean = np.array([104., 117., 124.])

    def read_image(self, path, verbose=False):
        '''
        画像を読み込んでクロップして返す
        arguments:
            path: 画像パス
        '''
        img = cv2.imread(path)
        if verbose:
            cv2.imshow("verbose: show image", img)
            cv2.waitKey(0)

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

    def one_hot(self, index):
        one_hot = np.zeros(self.num_classes)
        one_hot[index] = 1.0
        return one_hot

    def get_batches(self, n):
        '''
        ミニバッチを返す
        '''
        if len(self.batches) == 0:
            for i in xrange(0, len(self.image_paths), n):
                images = np.concatenate(map(self.read_image, self.image_paths[i:i+n]), axis=0)
                labels = self.labels[i:i+n]
                labels_index = [self.one_hot(self.list_labels.index(label)) for label in labels]
                self.batches.append((images, labels_index))
        random.shuffle(self.batches)
        return self.batches

    def __len__(self):
        return len(self.labels)

def test():
    image_input = ImageInput('./data/101Caltech_examples.txt')
    print "1st start"
    start = time.time()
    image_input.get_batches(10)
    end = time.time()
    print("1st done %f" % (end - start))
    print "2nd start"
    start = time.time()
    image_input.get_batches(10)
    end = time.time()
    print("2nd done %f" % (end - start))

if __name__ == '__main__':
    test()

