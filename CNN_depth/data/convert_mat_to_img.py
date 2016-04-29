#encoding: utf-8
import os
import numpy as np
import scipy
from scipy.io import loadmat
import h5py
import glob
from PIL import Image

import matplotlib.pyplot as plt

def convert_3d(directory):
    '''
    convet Make3D dataset to original and depth image
    '''
    print directory
    mat_list = glob.glob(directory)
    for mat in mat_list:
        mat_name = os.path.basename(mat)
        print("mat: %s" % (mat_name))
        mat_data = loadmat(mat)
        print mat_data['Position3DGrid'].shape
        print mat_data['Position3DGrid'][:, :, 3].shape
        print type(mat_data['Position3DGrid'][:, :, 3])
        img = mat_data['Position3DGrid'][:, :, 3]
        plt.subplot(1, 1, 1)
        plt.imshow(img)
        plt.title("Depth %s" % (mat_name))
        plt.show()


def convert_nyu(path, verbose=False):
    '''
    convet NYU Depth Dataset to original and depth image
    arguments:
        nyu depth dataset mat
    '''
    print("load dataset: %s" % (path))
    f = h5py.File(path)
    print f.keys()
    print f['images']
    print f['depths']
    for i, (image, depth) in enumerate(zip(f['images'], f['depths'])):
        ra_image = image.transpose(2, 1, 0)
        #print ra_image.shape
        ra_depth = depth.transpose(1, 0)
        #print ra_depth.shape
        re_depth = (ra_depth/np.max(ra_depth))*255.0
        image_pil = Image.fromarray(np.uint8(ra_image))
        depth_pil = Image.fromarray(np.uint8(re_depth))
        image_name = "nyu_datasets/%05d.jpg" % (i)
        image_pil.save(image_name)
        depth_name = "nyu_datasets/%05d.png" % (i)
        depth_pil.save(depth_name)
        depth_resize_name = "nyu_datasets/%05d_resize.png" % (i)
        depth_resize = depth_pil.resize((74, 55))
        depth_resize.save(depth_resize_name)
        depth_resize_array = np.asarray(depth_resize)
        #print depth_resize_array.shape

        if verbose:
            plt.subplot(121)
            plt.imshow(ra_image)
            plt.title("Original")
            plt.subplot(122)
            plt.imshow(ra_depth)
            plt.title("Depth")
            plt.show()


if __name__ == '__main__':
    current_directory = os.getcwd()
    path = os.path.join(current_directory + "/", "Train400Depth/*")
    nyu_path = 'nyu_depth_v2_labeled.mat'
    convert_nyu(nyu_path)
