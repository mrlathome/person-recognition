"""
Creates the necessary objects and executes functions of the system.
"""

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

    def get_sample(self):
        """
        Gets the latest sample from the input stream.
        :return: The sample image
        """
        image = self.data_acquisition.read_stream()
        return image
