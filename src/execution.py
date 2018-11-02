"""
Creates the necessary objects and executes functions of the system.
"""

import os
from copy import deepcopy

import cv2

from data_acquisition import CameStreamer
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
        self.cam_streamer = CameStreamer()

    def acquire_data(self):
        """
        Read data sets, process them, and create warehouses for storing them
        :return: None
        """
        trn_dir = os.path.join(self.pkg_dir, 'dataset', 'train')
        tst_dir = os.path.join(self.pkg_dir, 'dataset', 'test')
        self.data_acquisition.trn_wh = self.create_wh(trn_dir)
        self.data_acquisition.tst_wh = self.create_wh(tst_dir)
        trn_imgs = []
        tst_imgs = []
        for sample in self.data_acquisition.trn_wh.get_samples():
            trn_imgs.append(sample.image)
        for sample in self.data_acquisition.tst_wh.get_samples():
            tst_imgs.append(sample.image)
        trn_embs = self.model_engineering.encode(trn_imgs)
        tst_embs = self.model_engineering.encode(tst_imgs)
        for index, sample in enumerate(self.data_acquisition.trn_wh.get_samples()):
            sample.embedding = trn_embs[index]
        for index, sample in enumerate(self.data_acquisition.tst_wh.get_samples()):
            sample.embedding = tst_embs[index]

    def visualize(self, image, bbox):
        """
        Visualize a bounding box in an image
        :param sample: an image
        :return: the image overlaid with the bounding box
        """
        if bbox is None:
            return image
        _image = deepcopy(image)
        xmin, ymin, xmax, ymax = bbox
        start_pt = (xmin, ymin)
        end_pt = (xmax, ymax)
        color = (255, 0, 0)
        thickness = 2
        cv2.rectangle(_image, start_pt, end_pt, color)
        # thickness = 1
        # color = (255, 255, 255)
        # origin = (10, 100)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 1
        # cv2.putText(_image, str(sample.label), origin, font, font_scale, color, thickness, cv2.LINE_AA)
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
        face_image = image[ymin:ymax, xmin:xmax]
        face_image = self.data_processing.process(face_image)
        embedding = self.model_engineering.encode(face_image)
        uid = self.model_engineering.knn_classify(embedding)
        return uid

    def evaluate(self):
        samples_size = 0
        tp_size = 0
        for sample in self.data_acquisition.tst_wh.get_samples():
            detect_uid = self.id(sample.image)
            samples_size += 1
            if detect_uid == sample.uid:
                tp_size += 1

    def test(self):
        """
        Performing different tests
        :return: None
        """
        stop = False
        while not stop:
            image = self.cam_streamer.get_frame()
            if image is None:
                continue
            print('image.shape', image.shape)

            bbox = self.data_processing.detect_faces(image)
            # image = self.data_processing.crop(image, bbox)
            image = self.visualize(image, bbox)

            cv2.imshow('image', image)
            k = cv2.waitKey(30)
            if k == 27:  # wait for ESC key to exit
                self.cam_streamer.release()
                cv2.destroyAllWindows()
                stop = True
            elif k == ord('s'):  # wait for 's' key to save and exit
                cv2.imwrite('face.jpg', image)

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
                bbox = self.data_processing.detect_faces(image)
                image = self.data_processing.crop(image, bbox)
                image = self.data_processing.process(image)
                label = int(name_parts[0])
                sample.image = image
                sample.uid = label
                warehouse.add(sample)
        return warehouse
