"""
Test cases for data acquisition functions.
"""

import os
import unittest

from data_acquisition import DataAcquisition


class DataAcquisitionTestCase(unittest.TestCase):
    """
    Test methods in DataAcquisition
    """

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.pkg_dir = os.path.join('..')
        self.data_acquisition = DataAcquisition(self.pkg_dir)
        self.img_size = 160

    def test_create_wh(self):
        """
        The warehouse should be a dictionary of UID keys and record values
        :return: None
        """
        directory = os.path.join(self.pkg_dir, 'dataset', 'test')
        warehouse = self.data_acquisition.create_wh(directory)

        condition = len(warehouse.get_all()) > 0
        self.assertEqual(condition, True)


if __name__ == '__main__':
    unittest.main()
