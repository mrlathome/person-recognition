"""
Test cases for data preprocessing modul
"""
import cv2
import unittest
import tensorflow as tf

from data_processing import encode


class DataPreprocessingTestCase(unittest.TestCase):

    def test_encode(self):
        preprocessed_data = cv2.imread('images/andrew.jpg')
        encoded_image = encode(preprocessed_data)
        self.assertEqual(len(encoded_image) == 1, True)

    def test_preprocess(self):
        data = cv2.imread('images/andrew.jpg')


if __name__ == '__main__':
    unittest.main()
