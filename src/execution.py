"""
Creates the necessary objects and executes functions of the system.
"""

import numpy as np

from data_acquisition import DataAcquisition
from data_processing import DataProcessing
from model_engineering import ModelEngineering


class Execution:
    def __init__(self, pkg_dir):
        self.pkg_dir = pkg_dir
        self.data_acquisition = DataAcquisition(pkg_dir)
        self.data_processing = DataProcessing(self.pkg_dir)
        self.model_engineering = ModelEngineering(self.pkg_dir)

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

    def test(self):
        """
        Encode an image using the model for testing
        :return: The embedding of a test image
        """
        images, labels = self.data_acquisition.load_data('test', 160)
        test_image = np.array([images[2]])
        test_embedding = self.model_engineering.encode(test_image)
        return test_embedding
