"""
process data and produce valid output for other moduls
"""

import cv2
import tensorflow as tf

import align.detect_face


class DataProcessing:
    pnet_threshold = .6
    rnet_threshold = .7
    onet_threshold = .7

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
        gpu_memory_fraction = 1.0
        minsize = 50
        threshold = [DataProcessing.pnet_threshold, DataProcessing.rnet_threshold, DataProcessing.onet_threshold]
        factor = 0.709
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                    log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = align.detect_face.create_mtcnn(
                    sess, None)
                bounding_boxes, _ = align.detect_face.detect_face(
                    image, minsize, pnet,
                    rnet, onet, threshold, factor)
        if len(bounding_boxes) != 0:
            x1, y1, x2, y2, acc = bounding_boxes[0]
            return [int(x1), int(y1), int(x2), int(y2)]
        else:
            return None
