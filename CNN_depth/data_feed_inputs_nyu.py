#encoding: utf-8

import cv2
import numpy as np

import random
import h5py
from PIL import Image

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
        invalid_depths = []
        crop_size_h = FLAGS.crop_size_height
        crop_size_w = FLAGS.crop_size_width
        for i, (image, depth) in enumerate(zip(f['images'], f['depths'])):
            # transpose width and height and channel
            ra_image = image.transpose(2, 1, 0)
            
            # create crop information
            #img = ra_image.astype(np.float32)
            #h, w, c = img.shape
            #crop_height, crop_width = ((h-crop_size_h)/2, (w-crop_size_w)/2)
            # crop image
            #img = img[crop_height:crop_height+crop_size_h, crop_width:crop_width+crop_size_w, :]
            #img = img[None, ...]
            
            # resize image using PIL
            image_pil = Image.fromarray(np.uint8(ra_image))
            image_resize = image_pil.resize((crop_size_w, crop_size_h))
            image_array = np.asarray(image_resize)
            #print image_array.shape
            images.append(image_array)
            
            # transpose width and height
            ra_depth = depth.transpose(1, 0)
            
            # crop depth data
            # ra_depth = ra_depth[crop_height:crop_height+crop_size_h, crop_width:crop_width+crop_size_w]
            
            # resize depth data using PIL
            re_depth = (ra_depth/np.max(ra_depth))*255.0
            depth_pil = Image.fromarray(np.uint8(re_depth))
            # image resize (PIL 1st arg is widht, 2nd arg is height)
            target_depth = depth_pil.resize((74, 55))

            # add target as numpy array
            target_array = np.asarray(target_depth)
            #print target_array.shape
            depths.append(target_array[None, :])

            invalid_target_array = target_array.copy()
            #print invalid_target_array.shape
            #print np.max(invalid_target_array)
            #print np.min(invalid_target_array)
            invalid_target_array[invalid_target_array != 0] = 1
            #print np.max(invalid_target_array)
            #print np.min(invalid_target_array)
            invalid_depths.append(invalid_target_array[None, :])

        self.images = images
        self.depths = depths
        self.invalid_depths = depths
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
                invalid_labels = self.invalid_depths[i:i+n]
                self.batches.append((images, labels, invalid_labels))

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

