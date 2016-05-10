#encoding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.platform import gfile

from data import image_shape
from data import get_drive_dir, Calib, get_inds, image_shape, get_calib_dir

from PIL import Image

root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root_dir, 'data')

VELODYNE_DIRS = [
                "2011_09_26/2011_09_26_drive_0001_sync/velodyne_points",
                "2011_09_26/2011_09_26_drive_0002_sync/velodyne_points",
                "2011_09_26/2011_09_26_drive_0005_sync/velodyne_points"
                ]

def get_velodyne_points(velodyne_dir, frame):
    points_path = os.path.join(velodyne_dir, "data/%010d.bin" % frame)
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
    points = points[:, :3]  # exclude luminance
    return points

def load_disparity_points(velodyne_dir, frame, color=False, **kwargs):

    calib = Calib(color=color, **kwargs)

    # read velodyne points
    points = get_velodyne_points(velodyne_dir, frame)

    # remove all points behind image plane (approximation)
    points = points[points[:, 0] >= 1, :]

    # convert points to each camera
    xyd = calib.velo2disp(points)

    # take only points that fall in the first image
    xyd = calib.filter_disps(xyd)

    return xyd

def test(vel, verbose=False):
    velodyne_dir = os.path.join(data_dir, vel)

    velodyne_bin_dir = os.path.join(velodyne_dir, "data")
    bin_list = os.listdir(velodyne_bin_dir)

    for i in xrange(len(bin_list)):
        points = get_velodyne_points(velodyne_dir, i)
        print("points data type: %s" % type(points))
        print(points.shape)

        # x
        print("x max: %f, min: %f" % (np.max(points[:,0]), np.min(points[:,0])))
        # y
        print("y max: %f, min: %f" % (np.max(points[:,1]), np.min(points[:,1])))
        # z
        print("z max: %f, min: %f" % (np.max(points[:,2]), np.min(points[:,2])))

        # reflectance
        #print("r max: %f, min: %f" % (np.max(points[:,3]), np.min(points[:,3])))

        image_array = np.asarray([1224, 368])

        xyd = load_disparity_points(velodyne_dir, i, color=False)
        disp = np.zeros(image_shape, dtype=np.float)
        for x, y, d in np.round(xyd):
            disp[y, x] = d

        ones = np.ones(image_shape, dtype=np.float)

        image = Image.fromarray(np.uint8(ones - (disp/np.max(disp))*255.0))
        save_dir = os.path.join(velodyne_dir, "depth")
        if not gfile.Exists(save_dir):
            gfile.MakeDirs(save_dir)
        save_path = os.path.join(velodyne_dir, "depth/%010d.png" % i)
        image.save(save_path)

        if verbose:
            plt.subplot(121)
            plt.imshow(image)
            plt.title("Original")
            plt.subplot(122)
            plt.imshow(image)
            plt.title("Depth")
            plt.show()

            plt.figure(1)
            plt.clf()
            plt.imshow(disp)
            plt.show()


if __name__ == '__main__':
    for vel in VELODYNE_DIRS:
        test(vel)