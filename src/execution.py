"""
Creates the necessary objects and executes functions of the system.
"""

import os
from copy import deepcopy

from data_acquisition import DataAcquisition
from data_acquisition import Face
from data_acquisition import ImageSubscriber
from data_acquisition import Warehouse
from data_processing import DataProcessing
from model_engineering import ModelEngineering

import roslib
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage

roslib.load_manifest('person_recognition')

class Execution:
    def __init__(self, pkg_dir):
        self.pkg_dir = pkg_dir
        self.data_acquisition = DataAcquisition()
        self.data_processing = DataProcessing()
        self.model_engineering = ModelEngineering(self.pkg_dir)
        # self.cam_streamer = CamStreamer()
        self.image_subscriber = ImageSubscriber()
        self.acquire_data()
        self.model_engineering.knn_fit(self.data_acquisition.trn_wh)
        self.selected_face = None
        self.pub_img = rospy.Publisher('/person_recognition/image', CompressedImage, queue_size=1)
        self.pub_txt = rospy.Publisher('/person_recognition/crowd', String, queue_size=10)

    def talk(self, crowd_size):
        rospy.loginfo('Crowd size: {}'.format(crowd_size))
        self.pub_txt.publish(crowd_size)

    def publish_img(self, image):
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image)[1]).tostring()
        try:
            self.pub_img.publish(msg)
        except rospy.ROSInterruptException as e:
            rospy.loginfo('Could not publish an image.', e)

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

    def visualize(self, face):
        """
        Visualize a bounding box in an image
        :param sample: an image
        :return: the image overlaid with the bounding box
        """
        _image = deepcopy(face.container_image)
        xmin, ymin, xmax, ymax = face.bbox
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
        display_name = self.data_acquisition.trn_wh.get_name(face.uid)
        if display_name is None:
            display_name = str(face.uid)
        cv2.putText(_image, display_name, origin, font, font_scale, color, thickness, cv2.LINE_AA)
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

            frame = self.acquire_frame()
            if frame is not None:
                if self.selected_face is not None:
                    frame = self.visualize(self.selected_face)
                    self.publish_img(frame)
                
                """
                cv2.imshow('image', frame)
                k = cv2.waitKey(200)
                if k == 27:  # wait for ESC key to exit
                    cv2.destroyAllWindows()
                    stop = True
                elif k == ord('s'):  # wait for 's' key to save and exit
                    cv2.imwrite('face.jpg', frame)
                elif k == ord('a'):  # wait for 'a' key
                    self.add_person()
                elif k == ord('d'):  # wait for 'd' key
                    self.delete_person()
                """

    def delete_person(self, name=None):
        """
        Delete a Person from the training warehouse
        :param uid: the UID of the person
        :return: boolean value indicating the result
        """
        self.acquire_frame()
        if self.selected_face is not None and self.selected_face.uid != -1:
            uid = self.selected_face.uid
            images_paths = self.find_all_files(uid)
            for path in images_paths:
                os.remove(path)
            self.data_acquisition.trn_wh.delete(uid)
            self.model_engineering.knn_fit(self.data_acquisition.trn_wh)
            return True
        elif name is not None:
            self.data_acquisition.trn_wh.delete_by_name(name)
            return True
        else:
            return False

    def add_person(self, name=None):
        """
        Add a Face to the training warehouse
        :param image: the new face image
        :param uid: the UID of the face
        :return: boolean value indicating the result
        """
        self.acquire_frame()
        if self.selected_face is not None and self.selected_face.image is not None:
            image = self.selected_face.image
            uid = self.selected_face.uid
            image_path, new_uid = self.find_path(uid)
            cv2.imwrite(image_path, image)
            image = self.data_processing.process(image)
            face = Face(image, new_uid)
            face.name = name
            face.embedding = self.model_engineering.encode([face.image])
            self.data_acquisition.trn_wh.add(face)
            self.model_engineering.knn_fit(self.data_acquisition.trn_wh)
            return True
        else:
            return False

    def gendder_classifer(self):
        frame= self.image_subscriber.get_frame()
        print (frame.shape)
        bboxs = self.data_processing.detect_faces_bbox(frame)
        info, _ =self.model_engineering.gender.f_detector(frame,bboxs)
        return info


    def acquire_frame(self):
        """
        Retrieve a new frame from the camera stream and detect faces
        :return: None
        """
        frame = self.image_subscriber.get_frame()
        if frame is None or not frame.shape[0] > 0 or not frame.shape[1] > 0:
            return None
        bboxes = self.data_processing.detect_faces(frame)
        crowd_size = len(bboxes)
        self.talk(crowd_size)
        self.selected_face = None
        for bbox in bboxes:
            cropped_face = self.data_processing.crop(frame, bbox)
            face = self.data_processing.process(cropped_face)
            embedding = self.model_engineering.encode([face])
            uid = self.model_engineering.knn_classify(embedding[0])
            selected_image = cropped_face
            selected_uid = uid
            self.selected_face = Face(selected_image, selected_uid)
            self.selected_face.container_image = frame
            self.selected_face.bbox = bbox
        return frame

    def find_all_files(self, uid):
        """
        Find all of the files paths related to the query UID
        :param uid: the query UID
        :return: a list of paths
        """
        dataset_dir = os.path.join(self.pkg_dir, 'dataset', 'train')
        samples_num = 0
        for file in os.listdir(dataset_dir):
            name_parts = file.split('.')
            if name_parts[-1] == 'jpg':
                image_uid = name_parts[0]
                if int(image_uid) == uid:
                    samples_num += 1
        paths = []
        for sample in range(samples_num):
            del_face_uid = str(uid).zfill(4)
            del_face_number = str(sample).zfill(4)
            file_name = '{}.{}.jpg'.format(del_face_uid, del_face_number)
            image_path = os.path.join(dataset_dir, file_name)
            paths.append(image_path)
        return paths

    def find_path(self, uid):
        """
        Find an image path to save
        :param uid: the selected UID for the image
        :return: the image path
        """
        dataset_dir = os.path.join(self.pkg_dir, 'dataset', 'train')
        if uid == -1:
            persons = self.data_acquisition.trn_wh.get_persons()
            existing_uids = []
            for person in persons:
                existing_uids.append(person.uid)
            new_uid = 0
            while new_uid in existing_uids:
                new_uid += 1
            new_face_uid = str(new_uid).zfill(4)
            new_face_number = str(0).zfill(4)
        else:
            samples_num = 0
            for file in os.listdir(dataset_dir):
                name_parts = file.split('.')
                if name_parts[-1] == 'jpg':
                    image_uid = name_parts[0]
                    if int(image_uid) == uid:
                        samples_num += 1
            new_face_uid = str(uid).zfill(4)
            new_face_number = str(samples_num).zfill(4)
        file_name = '{}.{}.jpg'.format(new_face_uid, new_face_number)
        image_path = os.path.join(dataset_dir, file_name)
        return image_path, int(new_face_uid)

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
