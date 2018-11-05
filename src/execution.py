"""
Creates the necessary objects and executes functions of the system.
"""

import os
from copy import deepcopy

import cv2

from data_acquisition import CamStreamer
from data_acquisition import DataAcquisition
from data_acquisition import Face
from data_acquisition import Warehouse
from data_processing import DataProcessing
from model_engineering import ModelEngineering


class Execution:
    def __init__(self, pkg_dir):
        self.pkg_dir = pkg_dir
        self.data_acquisition = DataAcquisition()
        self.data_processing = DataProcessing()
        self.model_engineering = ModelEngineering(self.pkg_dir)
        self.cam_streamer = CamStreamer()
        self.acquire_data()
        self.model_engineering.knn_fit(self.data_acquisition.trn_wh)

    def acquire_data(self):
        """
        Read data sets, process them, and create warehouses for storing them
        :return: None
        """
        trn_dir = os.path.join(self.pkg_dir, 'dataset', 'train')
        tst_dir = os.path.join(self.pkg_dir, 'dataset', 'test')
        self.data_acquisition.trn_wh = self.create_wh(trn_dir)
        self.data_acquisition.tst_wh = self.create_wh(tst_dir)
        for face in self.data_acquisition.trn_wh.get_faces():
            face.embedding = self.model_engineering.encode([face.image])
        for face in self.data_acquisition.tst_wh.get_faces():
            face.embedding = self.model_engineering.encode([face.image])

    def visualize(self, image, bbox, uid):
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
        thickness = 1
        cv2.rectangle(_image, start_pt, end_pt, color, thickness)
        thickness = 1
        color = (255, 255, 255)
        origin = start_pt
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        cv2.putText(_image, str(uid), origin, font, font_scale, color, thickness, cv2.LINE_AA)
        return _image

    def id(self, image):
        """
        Identifies the person present in the image
        :param image: The input image
        :return: The UID of the person
        """
        # bbox = self.data_processing.detect_faces(image)
        # face_image = self.data_processing.crop(image, bbox)
        face_image = self.data_processing.process(image)
        embedding = self.model_engineering.encode([face_image])[0]
        uid = self.model_engineering.knn_classify(embedding)
        return uid

    def evaluate(self):
        """
        Evaluates the accuracy of the model on the test set
        :return: the true positive rate
        """
        accuracy = self.model_engineering.knn_eval(self.data_acquisition.tst_wh)
        return accuracy

    def test(self):
        """
        Performing different tests
        :return: None
        """
        stop = False
        while not stop:
            image = self.cam_streamer.get_frame()
            if image is None or not image.shape[0] > 0 or not image.shape[1] > 0:
                continue
            # print('image.shape', image.shape)

            bboxes = self.data_processing.detect_faces(image)
            for bbox in bboxes:
                face = self.data_processing.crop(image, bbox)
                face = self.data_processing.process(face)
                embedding = self.model_engineering.encode([face])
                uid = self.model_engineering.knn_classify(embedding[0])
                image = self.visualize(image, bbox, uid)

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
                image_path = os.path.join(directory, file)
                image = cv2.imread(image_path)
                bboxes = self.data_processing.detect_faces(image)
                for bbox in bboxes:
                    face = Face()
                    face_img = self.data_processing.crop(image, bbox)
                    face_img = self.data_processing.process(face_img)
                    label = int(name_parts[0])
                    face.image = face_img
                    face.uid = label
                    warehouse.add(face)
        return warehouse
