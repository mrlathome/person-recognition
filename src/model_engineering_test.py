#!/usr/bin/env python3

"""
test model_engineering module
"""

import os
import unittest

import cv2
import numpy as np

from data_acquisition import Sample
from model_engineering import ModelEngineering


class ModelEngineeringTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.pkg_dir = os.path.join('..')
        self.model_engineering = ModelEngineering(self.pkg_dir)
        self.img_size = 160

    def test_load_model(self):
        """
        After loading the model,
        the shape of input place holders should be (None, height, width, 3),
        the shape of phase train place holder should be (1),
        and the shape of the output place holder should be (None, 512)
        :return: None
        """
        frozen_graph_path = os.path.join(self.pkg_dir, 'InceptionResNetV1-VGGFace2', '20180402-114759.pb')
        imgs_ph, phase_train_ph, embs_ph, emb_size = self.model_engineering.load_model(frozen_graph_path)

        condition = emb_size == 512
        self.assertEqual(condition, True)

    def test_encode(self):
        """
        The size of each embedding should be 512
        :return: None
        """
        test_image_path = os.path.join(self.pkg_dir, 'dataset', 'test', '0000.0000.jpg')
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        test_image = cv2.resize(test_image, (self.img_size, self.img_size))
        test_image = np.array([test_image])
        test_embedding = self.model_engineering.encode(test_image)
        embedding_size = 512
        condition = test_embedding.shape[1] == embedding_size
        self.assertEqual(condition, True)

    def test_knn_classifiy(self):
        my_warehouse = [[], [], []]
        emb_Sample0 = [iris.data[0:30, :]]  # 0-49 for Iris-setosa
        emb_Sample1 = [iris.data[50:80, :]]  # 50-99 for Iris-versicolor
        emb_Sample2 = [iris.data[100:130, :]]  # 100-149 Iris-virginica

        for i in range(len(emb_Sample0[0])):
            sample0 = Sample()
            sample0.embedding = emb_Sample0[0][i]
            sample0.uid = 0
            my_warehouse[0].append(sample0)

        for i in range(len(emb_Sample1[0])):
            sample1 = Sample()
            sample1.embedding = emb_Sample0[0][i]
            sample1.uid = 1
            my_warehouse[1].append(sample1)

        for i in range(len(emb_Sample1[0])):
            sample2 = Sample()
            sample2.embedding = emb_Sample0[0][i]
            sample2.uid = 2
            my_warehouse[2].append(sample2)
            query = iris.data[132:133, :]
            print
            self.model_engineering.knn_classify(warehouse=my_warehouse, query_emb=query)


if __name__ == '__main__':
    unittest.main()
