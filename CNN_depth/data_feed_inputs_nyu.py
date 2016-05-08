#encoding: utf-8

from tensorflow.python.platform import gfile

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
        images_depths_invalid = []
        images_depths_invalid_test = []
        images_depths_invalid_val = []
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

            # add target as numpy array and change to 0~1
            target_array = np.asarray(target_depth)/255.0
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

            # append summary array
            cnum = random.random()
            if cnum <= 0.08:
                # for test
                images_depths_invalid_test.append((image_array, target_array[None, :], invalid_target_array[None, :]))
            elif 0.08 < cnum and cnum < 0.16:
                # for validataion
                images_depths_invalid_val.append((image_array, target_array[None, :], invalid_target_array[None, :]))
            else:
                # for train
                images_depths_invalid.append((image_array, target_array[None, :], invalid_target_array[None, :]))
 

        self.images = images
        self.depths = depths
        self.images_depths_invalid = images_depths_invalid
        self.images_depths_invalid_test = images_depths_invalid_test
        self.images_depths_invalid_val = images_depths_invalid_val

        print("num of train: %d" % (len(images_depths_invalid)))
        print("num of test: %d" % (len(images_depths_invalid_test)))
        if not gfile.Exists("test_set"):
            gfile.MakeDirs("test_set")
        for i, test_img in enumerate(images_depths_invalid_test):
            #ra_img = (test_img / np.max(test_img)) * 255.0
            img_pil = Image.fromarray(np.uint8(test_img[0]))
            img_name = "%s/%05d.png" % ("test_set", i)
            img_pil.save(img_name)

        print("num of validation: %d" % (len(images_depths_invalid_val)))
        if not gfile.Exists("validation_set"):
            gfile.MakeDirs("validation_set")
        for i, val_img in enumerate(images_depths_invalid_val):
            #ra_img = (val_img / np.max(val_img)) * 255.0
            img_pil = Image.fromarray(np.uint8(val_img[0]))
            img_name = "%s/%05d.png" % ("validation_set", i)
            img_pil.save(img_name)

        self.invalid_depths = depths
        self.batches = []
 
    def get_batches(self, n):
        '''
        ミニバッチを返す
        '''
        print("shuffle start.")
        random.shuffle(self.images_depths_invalid)
        random.shuffle(self.images_depths_invalid)
        print("shuffle done.")

        self.batches = []
        for i in xrange(0, len(self.images_depths_invalid), n):
            if i % 500 == 0:
                print("generate batches: size %d, index %d" % (n, i))
            images_depths_invalid = self.images_depths_invalid[i:i+n]
            images = [a[0] for a in images_depths_invalid]
            labels = [a[1] for a in images_depths_invalid]
            invalid_labels = [a[2] for a in images_depths_invalid]
            self.batches.append((images, labels, invalid_labels))

        #print("batch shuffle start")
        # 2回shuffle
        #random.shuffle(self.batches)
        #random.shuffle(self.batches)
        #print("batch shuffle done")
        print("the number of batches: %d" % (len(self.batches)))
        return self.batches

    def get_test(self):
        return self.images_depths_invalid_test

    def get_validation(self, shuffle=False):
        if shuffle:
            print("shuffle start.")
            random.shuffle(self.images_depths_invalid_val)
            random.shuffle(self.images_depths_invalid_val)
            print("shuffle done.")
        images = [a[0] for a in self.images_depths_invalid_val]
        labels = [a[1] for a in self.images_depths_invalid_val]
        invalid_labels = [a[2] for a in self.images_depths_invalid_val]
        return (images, labels, invalid_labels)

    def __len__(self):
        return len(self.labels)


def output_predict(depths, output_dir):
    if not gfile.Exists(output_dir):
        gfile.MakeDirs(output_dir)

    print("the number of output predict: %d" % len(depths))
    for i, depth in enumerate(depths):
        depth = depth.transpose(2, 0, 1)
        ra_depth = (depth/np.max(depth))*255.0
        depth_pil = Image.fromarray(np.uint8(ra_depth[0]), mode="L")
        depth_name = "%s/%05d.png" % (output_dir, i)
        depth_pil.save(depth_name)


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

