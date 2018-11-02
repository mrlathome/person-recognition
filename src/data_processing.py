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
        self.fn_size = 0
        self.processed_samples_size = 0
        self.graph = tf.Graph()
        self.graph.as_default()
        self.session = tf.Session()
        self.session.as_default()
        self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(self.session, None)

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
        minsize = 50
        threshold = [self.pnet_threshold, self.rnet_threshold, self.onet_threshold]
        factor = 0.709
        bounding_boxes, _ = align.detect_face.detect_face(image, minsize, self.pnet, self.rnet, self.onet, threshold,
                                                          factor)
        self.processed_samples_size += 1
        print('Processed samples:', self.processed_samples_size)
        if len(bounding_boxes) != 0:
            x1, y1, x2, y2, acc = bounding_boxes[0]
            return [int(x1), int(y1), int(x2), int(y2)]
        else:
            self.fn_size += 1
            print('False negatives:', self.fn_size)
            return None

    def crop(self, image, bbox):
        """
        Crop an image according to a bounding box
        :param bbox: the bounding box
        :return: the cropped image
        """
        if not bbox:
            return image
        xmin, ymin, xmax, ymax = bbox
        cropped = image[ymin:ymax, xmin:xmax]
        return cropped
