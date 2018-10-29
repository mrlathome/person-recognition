"""
Creates the necessary objects and executes functions of the system.
"""

from copy import deepcopy

import cv2

from data_acquisition import DataAcquisition
from data_processing import DataProcessing
from model_engineering import ModelEngineering


class Execution:
    def __init__(self, pkg_dir):
        self.pkg_dir = pkg_dir
        self.data_acquisition = DataAcquisition(pkg_dir)
        self.data_processing = DataProcessing(self.pkg_dir)
        self.model_engineering = ModelEngineering(self.pkg_dir)
        for sample in self.data_acquisition.warehouse.get_all():
            sample.embedding = self.model_engineering.encode([sample.image])

    def visualize(self, sample):
        """
        Visualize a sample using its attributes
        :param sample: a sample in a dataset
        :return: image overlaid by the related information
        """
        _image = deepcopy(sample.image)
        # thickness = 1
        # color = (255, 255, 255)
        # origin = (10, 100)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 1
        # cv2.putText(_image, str(sample.label), origin, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.imshow('UID: {}'.format(sample.uid), _image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return _image

    def id(self, image):
        """
        Identifies the person present in the image
        :param image: The input image
        :return: The UID of the person
        """
        faces_bboxes = self.data_processing.detect_faces(image)
        uids = []
        for bbox in faces_bboxes:
            xmin, xmax, ymin, ymax = bbox
            encoding = self.model_engineering.encode(image[xmin:xmax, ymin, ymax])
            uid = self.model_engineering.knn_classify(encoding)
            uids.append(uid)
        return uids

    def evaluate(self):
        pass

    def test(self):
        """
        Encode an image using the model for testing
        :return: The embedding of a test image
        """
        sample = self.data_acquisition.warehouse.get_all()[0]
        self.visualize(sample)
        return sample.embedding
