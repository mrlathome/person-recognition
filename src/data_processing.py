"""
process data and produce valid output for other moduls
"""

import numpy as np
import tensorflow as tf
from scipy import misc

import align.detect_face


class DataProcessing:
    def __init__(self):
        self.face_crop_size = 160
        self.fn_size = 0
        self.processed_samples_size = 0
        self.graph = tf.Graph()
        self.graph.as_default()
        self.session = tf.Session()
        self.session.as_default()
        self.pnet_threshold = .6
        self.rnet_threshold = .7
        self.onet_threshold = .7
        self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(self.session, None)
        self.face_crop_margin = 32

    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    def process(self, image):
        """
        Process the input image
        :param image: the input image
        :return: the processed image
        """
        # Resize the image
        # image = cv2.resize(image, (self.img_size, self.img_size))
        # image = misc.imresize(image, (self.face_crop_size, self.face_crop_size), interp='bilinear')
        image = self.prewhiten(image)
        return image

    def detect_faces(self, image):
        """
        Detects faces in the input image and returns a list of bounding boxes
        corresponding to the present faces
        :param image: The input image
        :return: A list of bounding boxes
        """
        minsize = 50
        threshold = [self.pnet_threshold, self.rnet_threshold, self.onet_threshold]
        factor = 0.709
        bounding_boxes, _ = align.detect_face.detect_face(image, minsize, self.pnet, self.rnet, self.onet, threshold,
                                                          factor)
        self.processed_samples_size += 1
        # print('Processed samples:', self.processed_samples_size)
        if len(bounding_boxes) != 0:
            x1, y1, x2, y2, acc = bounding_boxes[0]
            return [int(x1), int(y1), int(x2), int(y2)]
        else:
            self.fn_size += 1
            # print('False negatives:', self.fn_size)
            return None

    def crop(self, image, bbox):
        """
        Crop an image according to a bounding box
        :param bbox: the bounding box
        :return: the cropped image
        """
        if not bbox:
            bbox = [0, 0, np.maximum(image.shape[0] - 1, 0), np.maximum(image.shape[1] - 1, 0)]
        bounding_box = np.zeros(4, dtype=np.int32)
        img_size = np.asarray(image.shape)[0:2]
        bounding_box[0] = np.maximum(bbox[0] - self.face_crop_margin / 2, 0)
        bounding_box[1] = np.maximum(bbox[1] - self.face_crop_margin / 2, 0)
        bounding_box[2] = np.minimum(bbox[2] + self.face_crop_margin / 2, img_size[1])
        bounding_box[3] = np.minimum(bbox[3] + self.face_crop_margin / 2, img_size[0])
        cropped = image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :]
        image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')
        return image
