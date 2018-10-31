"""
Creates the necessary objects and executes functions of the system.
"""

import os
from copy import deepcopy

import cv2

from data_acquisition import DataAcquisition
from data_acquisition import Sample
from data_acquisition import Warehouse
from data_processing import DataProcessing
from model_engineering import ModelEngineering


class Execution:
    def __init__(self, pkg_dir):
        self.pkg_dir = pkg_dir
        self.data_acquisition = DataAcquisition()
        self.data_processing = DataProcessing()
        self.model_engineering = ModelEngineering(self.pkg_dir)
        self.acquire_data()

    def acquire_data(self):
        """
        Read data sets, process them, and create warehouses for storing them
        :return: None
        """
        trn_dir = os.path.join(self.pkg_dir, 'dataset', 'train')
        tst_dir = os.path.join(self.pkg_dir, 'dataset', 'test')
        self.data_acquisition.trn_wh = self.create_wh(trn_dir)
        self.data_acquisition.tst_wh = self.create_wh(tst_dir)
        for sample in self.data_acquisition.trn_wh.get_samples():
            sample.embedding = self.model_engineering.encode([sample.image])
        for sample in self.data_acquisition.tst_wh.get_samples():
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
        face_bbox = self.data_processing.detect_faces(image)
        # In case there isn't any face in the image
        if not face_bbox:
            return -1
        xmin, ymin, xmax, ymax = face_bbox
        embedding = self.model_engineering.encode(image[ymin:ymax, xmin:xmax])
        uid = self.model_engineering.knn_classify(embedding)
        return uid

    def evaluate(self):
        pass

    def test(self):
        """
        Encode an image using the model for testing
        :return: The embedding of a test image
        """
        sample = self.data_acquisition.tst_wh.get_all()[0]
        self.visualize(sample)
        return sample.embedding

    def create_wh(self, directory):
        """
        Read a data set and create a new warehouse
        :param directory: the directory of the data set
        :return: the warehouse containing the data set
        """
        warehouse = Warehouse()
        for file in os.listdir(directory):
            name_parts = file.split('.')
            if name_parts[-1] == 'jpg':
                sample = Sample()
                image_path = os.path.join(directory, file)
                image = cv2.imread(image_path)
                image = self.data_processing.process(image)
                label = int(name_parts[0])
                sample.image = image
                sample.uid = label
                warehouse.add(sample)
        return warehouse
