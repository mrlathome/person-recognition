"""
Acquires data from multiple sources
"""

import matplotlib.image
import os

import numpy as np
import scipy.misc


class DataAcquisition:
    def __init__(self, pkg_dir):
        self.pkg_dir = pkg_dir

    def load_data(self, name, img_size):
        """
        Read a dataset of images
        :param name: train/test set directory name
        :param img_size: image size
        :return: numpy arrays of images and labels
        """

        imgs_dir = os.path.join(self.pkg_dir, 'dataset', name)
        images = []
        labels = []
        for file in os.listdir(imgs_dir):
            name_parts = file.split('.')
            if name_parts[-1] == 'jpg':
                img_path = os.path.join(imgs_dir, file)
                img = matplotlib.image.imread(img_path)
                img = scipy.misc.imresize(img, (img_size, img_size))
                images.append(img)
                label = int(name_parts[0])
                labels.append(label)

        images = np.array(images)
        labels = np.array(labels)

        return images, labels


def load_img(file):
    """
    Reads image from disk
    :param file: image file path
    :return: RGB image as numpy array
    """
    # Creating an empty array only to see the test fail
    # img = np.array([])
    # Load image file from disk
    img = image.imread(file)
    return img
