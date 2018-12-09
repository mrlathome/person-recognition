"""
Acquire data from multiple sources
"""
import cv2
from sensor_msgs.msg import CompressedImage
import numpy as np
import rospy


class Face:
    def __init__(self, image=None, uid=None):
        """
        Initialize the Face
        :param image: the image
        :param uid: the UID
        """
        self.image = image
        self.uid = uid
        # The 512-vector corresponding to the encoded image
        self.embedding = None
        self.name = None
        self.bbox = None
        self.container_image = None


class Person:
    def __init__(self, face):
        """
        Initialize the Person
        :param face: the first Face
        """
        self.uid = face.uid
        self.name = face.name
        self.faces = []
        self.add(face)

    def add(self, face):
        """
        Add a face to the a Person
        :param face: the new Face
        :return: None
        """
        self.faces.append(face)

    def delete(self, face):
        """
        Delete a Face from a Person
        :param face: the Face to be deleted
        :return: None
        """
        if face in self.faces:
            self.faces.remove(face)


class Warehouse:
    def __init__(self):
        self.persons = {}

    def add(self, face):
        """
        Add a new Face
        :param face: the Face
        :return: None
        """
        if face.uid in self.persons.keys():
            self.persons[face.uid].add(face)
        else:
            self.persons[face.uid] = Person(face)

    def delete(self, uid):
        """
        Delete an existing Person
        :param uid: the UID
        :return: None
        """
        if uid in self.persons.keys():
            self.persons.pop(uid)

    def delete_by_name(self, name):
        """
        Delete an existing Person
        :param uid: the name
        :return: None
        """
        for person in self.persons.values():
            if person.name == name:
                self.delete(person.uid)
                return

    def get(self, uid):
        """
        Retrieve a Person by UID
        :param uid: the query UID
        :return: Faces of the Person
        """
        for person in self.persons:
            if person.uid == uid:
                return person

    def get_faces(self):
        """
        Retrieve every existing Face
        :return: a list of Faces
        """
        faces = []
        for person in self.persons.values():
            for face in person.faces:
                faces.append(face)
        return faces

    def get_persons(self):
        """
        Retrieve every existing Person
        :return: a list of Persons
        """
        persons = []
        for person in self.persons.values():
            persons.append(person)
        return persons

    def get_name(self, uid):
        """
        Return the name of the Person related to the UID
        :param uid: the UID
        :return: the name
        """
        for person in self.persons.values():
            if person.uid == uid:
                return person.name


class DataAcquisition:
    def __init__(self):
        self.trn_wh = None
        self.tst_wh = None


class CamStreamer:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)

    def get_frame(self):
        """
        Capture the latest frame
        :return: the frame
        """
        ret, frame = self.cap.read()
        if not ret:
            print('Failed to read a frame.')
            return None
        return frame

    def release(self):
        """
        Release the capture when everything is done
        :return: None
        """
        self.cap.release()


class ImageSubscriber:
    def __init__(self):
        self.frame = None
        self.topic = '/camera/rgb/image_rect_color/compressed'
        self.subscriber = rospy.Subscriber(self.topic, CompressedImage, self.callback, queue_size=1)

    def callback(self, ros_data):
        np_arr = np.fromstring(ros_data.data, np.uint8)
        self.frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def get_frame(self):
        """
        Capture the latest frame
        :return: the current frame
        """
        return self.frame
