"""
Test cases for data acquisition functions.
"""

import os
import unittest

from src.data_acquisition import load_img


class DataAcquisitionTestCase(unittest.TestCase):
    """
    Tests the data acquisition.
    """

    def test_load_img(self):
        """
        The returned image should be a numpy array of dimension 3
        :return: None
        """

        file_path = os.path.join('..', 'images', 'andrew.jpg')
        image = load_img(file_path)

        condition = image.ndim == 3
        self.assertEqual(condition, True)


if __name__ == '__main__':
    unittest.main()
