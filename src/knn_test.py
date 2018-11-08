#!/usr/bin/env python3

import unittest

import numpy as np

from knn import KNN


class KNNTestCase(unittest.TestCase):
    """
    Test cases for the KNN implementation
    """

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.knn = KNN(k=5)
        self.train_data = np.array(
            [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2],
             [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4.0, 1.3]])
        self.train_label = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        self.test_data = np.array(
            [[5.0, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3],
             [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9]])
        self.test_label = np.array([0, 0, 1, 1, 2, 2])

    def test_fit(self):
        """
        The return value of the function should be equal to the number of the classes in the data set
        :return: None
        """
        num_classes = self.knn.fit(self.test_data, self.test_label)
        condition = num_classes == 3
        self.assertEqual(condition, True)

    def test_compute_distance(self):
        """
        The distance between the two input vectors should be a floating point value inside the zero to one interval
        :return: None
        """
        sample0 = self.train_data[0]
        sample1 = self.train_data[1]
        distance = self.knn.compute_distance(sample0, sample1)
        condition = 0.0 <= distance <= 1.0
        self.assertEqual(condition, True)

    def test_get_neighbours(self):
        """
        The returned neighbours should be a list of tuples, each of which contains the label and the distance
        :return: None
        """
        self.knn.fit(self.train_data, self.train_label)
        query = self.test_data[0]
        neighbours = self.knn.get_neighbors(query)
        condition0 = len(neighbours) == self.knn.k
        condition1 = len(neighbours[0]) == 2
        condition = condition0 and condition1
        self.assertEqual(condition, True)

    def test_classify(self):
        """
        The classified label of the normal sample should correspond to its ground truth label
        and for the anomaly sample that does not belong to any of the training classes it should be equal to -1
        :return: None
        """
        self.knn.fit(self.train_data, self.train_label)
        normal_data = self.test_data[0]
        normal_label = self.test_label[0]
        anomaly_data = self.test_data[-1]
        normal_pred = self.knn.classify(normal_data)
        anomaly_pred = self.knn.classify(anomaly_data)
        condition0 = normal_pred == normal_label
        condition1 = anomaly_pred == -1
        condition = condition0 and condition1
        self.assertEqual(condition, True)

    def test_evaluate(self):
        """
        The returned accuracy should be equal to 1.0 in the case where the training and test set are the same
        and in the case where the training set and test set are different it should be in the interval zero to one
        :return: None
        """
        self.knn.fit(self.train_data, self.train_label)
        accuracy_perfect = self.knn.evaluate(self.train_data, self.train_label)
        condition0 = accuracy_perfect == 1.0
        accuracy_imperfect = self.knn.evaluate(self.test_data, self.test_label)
        condition1 = 0.0 < accuracy_imperfect < 1.0
        condition = condition0 and condition1
        self.assertEqual(condition, True)


if __name__ == '__main__':
    unittest.main()
