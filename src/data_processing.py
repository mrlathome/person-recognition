"""
process data and produce valid output for other moduls
"""

import numpy as np


class DataProcessing:
    def __init__(self, pkg_dir):
        self.pkg_dir = pkg_dir

    def detect_faces(self, image):
        """
        Detects faces in the input image and returns a list of bounding boxes
        corresponding to the present faces
        :param image: The input image
        :return: A list of bounding boxes
        """
        return []
