#encoding: utf-8

import cv2
import numpy as np

import random
import h5py

import time
# settings
import settings
FLAGS = settings.FLAGS

class ImageInput(object):
    def __init__(self, nyu_mat_path):
        '''
        初期化
        arguments:
            NYU Datasets V2 mat path
        '''
        print("load dataset: %s" % (nyu_mat_path))
        f = h5py.File(nyu_mat_path)
        images = []
        depths = []
        for i, (image, depth) in enumerate(zip(f['images'], f['depths'])):
            ra_image = image.transpose(2, 1, 0)
            images.append(ra_image)
            ra_depth = depth.transpose(1, 0)
            depths.append(ra_depth)
        self.images = images
        self.depths = depths
        self.batches = []
 
    def get_batches(self, n):
        '''
        ミニバッチを返す
        '''
        if len(self.batches) == 0:
            for i in xrange(0, len(self.images), n):
                if i % 500 == 0:
                    print("generate batches: size %d, index %d" % (n, i))
                images = self.images[i:i+n]
                labels = self.depths[i:i+n]
                self.batches.append((images, labels))

        print("shuffle start")
        # 2回shuffle
        random.shuffle(self.batches)
        random.shuffle(self.batches)
        print("shuffle done")
        print("the number of batches: %d" % (len(self.batches)))
        return self.batches

    def __len__(self):
        return len(self.labels)


def test():
    image_input = ImageInput('./data/nyu_depth_v2_labeled.mat')
    print "1st start"
    start = time.time()
    batches = image_input.get_batches(5)
    print len(batches)
    end = time.time()
    print("1st done %f" % (end - start))
    print "2nd start"
    start = time.time()
    image_input.get_batches(5)
    end = time.time()
    print("2nd done %f" % (end - start))

if __name__ == '__main__':
    test()

