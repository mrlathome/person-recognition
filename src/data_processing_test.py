"""
Test cases for data preprocessing modul
"""
import unittest

import cv2
import numpy as np

from data_processing import DataProcessing


class DataPreprocessingTestCase(unittest.TestCase):

    def test_detect_faces(self):
        """
        test the detect_dace function in data_processing module
        :return: None
        """
        face_img = cv2.imread('../dataset/test/0000.0000.jpg')
        car_imag = np.random.randint(255, size=(160, 160, 3), dtype=np.uint8)
        data_processing = DataProcessing()
        print(type(face_img))
        bounding_box_1 = data_processing.detect_faces(face_img)
        bounding_box_2 = data_processing.detect_faces(car_imag)
        condition_1 = len(bounding_box_1) == 1 and len(bounding_box_1[0]) == 4
        condition_2 = len(bounding_box_2) == 0
        condition = condition_1 and condition_2
        self.assertEqual(condition, True)


if __name__ == '__main__':
    unittest.main()
