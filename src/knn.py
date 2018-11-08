"""
kNN implemented from Scratch in Python
"""
import operator

import numpy as np


class KNN:
    def __init__(self, k=5):
        """
        Define the training data set and initialize the K parameter
        :param k: The parameter K
        """
        self.train_data = None
        self.train_label = None
        self.k = k

    def fit(self, train_data, train_label):
        """
        Fit to the training data set
        :param train_data: training data
        :param train_label: training labels
        :return: the number of classes
        """
        self.train_data = train_data
        self.train_label = train_label
        labels = np.unique(self.train_label)
        num_classes = len(labels)
        return num_classes

    def compute_distance(self, sample0, sample1):
        """
        Compute the distance between two samples
        :param sample0: the first sample
        :param sample1: the second sample
        :return: the distance
        """
        distance = np.linalg.norm(sample0 - sample1)
        return distance

    def get_neighbors(self, query):
        """
        Get K nearest neighbours to the query data
        :param query: the query data
        :return: the neighbours
        """
        neighbours = []
        for data, label in zip(self.train_data, self.train_label):
            distance = self.compute_distance(query, data)
            neighbours.append((label, distance))
        neighbours.sort(key=lambda kv: kv[1])
        return neighbours[:self.k]

    def classify(self, query):
        """
        Classify the query data
        :param query: the query data
        :return: the classified label
        """
        label = -1
        neighbours = self.get_neighbors(query)
        if neighbours[0][1] > 0.7:
            return label
        class_votes = {}
        for neighbour in neighbours:
            neighbour_label = neighbour[0]
            neighbour_distance = neighbour[1]
            if neighbour_distance == 0:
                score = 1.0
            else:
                score = 1.0 / neighbour_distance
            if neighbour_label in class_votes.keys():
                class_votes[neighbour_label] += score
            else:
                class_votes[neighbour_label] = score
        label = max(class_votes.items(), key=operator.itemgetter(1))[0]
        return label

    def evaluate(self, test_data, test_labels):
        """
        Evaluate the model on a test set
        :param test_data: test data
        :param test_labels: test labels
        :return: the accuracy
        """
        samples_size = len(test_labels)
        true_positives = 0
        for data, label in zip(test_data, test_labels):
            pred_label = self.classify(data)
            if pred_label == label:
                true_positives += 1
        accuracy = true_positives / samples_size
        return accuracy
