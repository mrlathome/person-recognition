"""
Acquire data from multiple sources
"""


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


class Person:
    def __init__(self, sample):
        """
        Initialize the Person
        :param sample: the first sample
        """
        self.uid = sample.uid
        self.name = None
        self.samples = []
        self.add(sample)

    def add(self, sample):
        """
        Add a sample to the a Person
        :param sample: the new sample
        :return: None
        """
        self.samples.append(sample)

    def delete(self, sample):
        """
        Delete a sample from a Person
        :param sample: the sample to be deleted
        :return: None
        """
        if sample in self.samples:
            self.samples.remove(sample)


class Warehouse:
    def __init__(self):
        self.persons = {}

    def add(self, sample):
        """
        Add a new sample
        :param sample: the sample
        :return: None
        """
        if sample.uid in self.persons.keys():
            self.persons[sample.uid].add(sample)
        else:
            self.persons[sample.uid] = Person(sample)

    def delete(self, uid):
        """
        Delete an existing Person
        :param uid: the UID
        :return: None
        """
        if uid in self.persons.keys():
            self.persons.pop(uid)

    def get(self, uid):
        """
        Retrieve a Person by UID
        :param uid: the query UID
        :return: samples of the Person
        """
        for person in self.persons:
            if person.uid == uid:
                return person

    def get_samples(self):
        """
        Retrieve every existing sample
        :return: a list of samples
        """
        samples = []
        for person in self.persons.values():
            for sample in person.samples:
                samples.append(sample)
        return samples

    def get_persons(self):
        """
        Retrieve every existing Person
        :return: a list of Persons
        """
        persons = []
        for person in self.persons.values():
            persons.append(person)
        return persons


class DataAcquisition:
    def __init__(self):
        self.trn_wh = None
        self.tst_wh = None
