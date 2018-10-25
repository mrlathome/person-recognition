"""
Acquires data from multiple sources
"""

import os

import cv2
import matplotlib.image
import numpy as np


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
                image_path = os.path.join(imgs_dir, file)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (img_size, img_size))
                images.append(image)
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
    img = matplotlib.image.imread(file)
    return img
