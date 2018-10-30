"""
process data and produce valid output for other moduls
"""

import os
import sys

import align.detect_face
import cv2
import tensorflow as tf


class DataProcessing:
    def __init__(self):
        self.img_size = 160

    def process(self, image):
        """
        Process the input image
        :param image: the input image
        :return: the processed image
        """
        # Resize the image
        image = cv2.resize(image, (self.img_size, self.img_size))
        return image

    def detect_faces(self, image):
        """
        Detects faces in the input image and returns a list of bounding boxes
        corresponding to the present faces
        :param image: The input image
        :return: A list of bounding boxes
        """
        sys.path.append('/home/pooya/Documents/code/facenet-master/src')
        gpu_memory_fraction = 1.0
        minsize = 50
        threshold = [0.6, 0.7, 0.7]
        factor = 0.709
        image_dir = ''
        images = os.listdir(image_dir)
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                    log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = align.detect_face.create_mtcnn(
                    sess, None)
            for i in images:
                img = cv2.imread(os.path.expanduser(image_dir + i))
                bounding_boxes, _ = align.detect_face.detect_face(
                    img, minsize, pnet,
                    rnet, onet, threshold, factor)

        return [int(x1), int(y1), int(x1 + w), int(y1 + h)]
