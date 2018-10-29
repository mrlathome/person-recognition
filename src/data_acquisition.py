"""
Acquires data from multiple sources
"""

import os

import cv2


class Sample:
    def __init__(self, image=None, uid=None):
        """
        Initialize the sample
        :param image: the image
        :param uid: the UID
        """
        self.image = image
        self.uid = uid
        # The 512-vector corresponding to the encoded image
        self.embedding = None
        self.name = None


class Record:
    def __init__(self, sample):
        """
        Initialize the record
        :param uid: the UID
        """
        self.uid = sample.uid
        self.name = None
        self.samples = []
        self.samples.append(sample)

    def add(self, sample):
        """
        Add a sample to the a record
        :param sample: the new sample
        :return: None
        """
        self.samples.append(sample)

    def delete(self, sample):
        """
        Delete a sample from a record
        :param sample: the sample to be deleted
        :return: None
        """
        if sample in self.samples:
            self.samples.remove(sample)


class Warehouse:
    def __init__(self):
        self.records = {}

    def add(self, sample):
        """
        Add a new sample
        :param sample: the sample
        :return: None
        """
        if sample.uid in self.records.keys():
            self.records[sample.uid].add(sample)
        else:
            self.records[sample.uid] = Record(sample)

    def delete(self, uid):
        """
        Delete an existing record
        :param uid: the UID
        :return: None
        """
        if uid in self.records.keys():
            self.records.pop(uid)

    def get(self, uid):
        """
        Retrieve a record by UID
        :param uid: the query UID
        :return: a record of samples
        """
        for record in self.records:
            if record.uid == uid:
                return record

    def get_all(self):
        """
        Retrieve every existing sample
        :return: a list of samples
        """
        samples = []
        for record in self.records.values():
            for sample in record.samples:
                samples.append(sample)
        return samples


class DataAcquisition:
    def __init__(self, pkg_dir):
        self.pkg_dir = pkg_dir
        self.img_size = 160
        trn_dir = os.path.join(self.pkg_dir, 'dataset', 'train')
        tst_dir = os.path.join(self.pkg_dir, 'dataset', 'test')
        self.trn_wh = self.load(trn_dir)
        self.tst_wh = self.load(tst_dir)

    def load(self, directory):
        """
        Read a data set and create a new warehouse
        :param directory: the directory of the dataset
        :return: the warehouse of the data set
        """
        warehouse = Warehouse()
        for file in os.listdir(directory):
            name_parts = file.split('.')
            if name_parts[-1] == 'jpg':
                sample = Sample()
                image_path = os.path.join(directory, file)
                image = cv2.imread(image_path)
                image = cv2.resize(image, (self.img_size, self.img_size))
                label = int(name_parts[0])
                sample.image = image
                sample.uid = label
                warehouse.add(sample)
        return warehouse
