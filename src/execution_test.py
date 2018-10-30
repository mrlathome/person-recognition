"""
Test cases for execution functions.
"""

import os
import unittest

from execution import Execution


class ExecutionTestCase(unittest.TestCase):
    """
    Test methods in Execution
    """

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.pkg_dir = os.path.join('..')
        self.execution = Execution(self.pkg_dir)
        self.img_size = 160

    def test_create_wh(self):
        """
        The warehouse should be a dictionary of UID keys and Person values
        :return: None
        """
        directory = os.path.join(self.pkg_dir, 'dataset', 'test')
        warehouse = self.execution.create_wh(directory)

        condition = len(warehouse.get_samples()) > 0
        self.assertEqual(condition, True)

    def test_acquire_data(self):
        """
        This function should create two warehouses and populate them with samples
        :return: None
        """
        self.execution.acquire_data()
        condition0 = len(self.execution.data_acquisition.trn_wh.get_samples()) > 0
        condition1 = len(self.execution.data_acquisition.tst_wh.get_samples()) > 0
        condition = condition0 and condition1
        self.assertEqual(condition, True)


if __name__ == '__main__':
    unittest.main()
