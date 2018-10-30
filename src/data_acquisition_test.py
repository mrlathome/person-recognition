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
        self.data_acquisition = DataAcquisition()


if __name__ == '__main__':
    unittest.main()
