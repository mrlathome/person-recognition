"""
process data and produce valid output for other moduls
"""

import cv2


class DataProcessing:
    def __init__(self):
        self.img_size = 160

    def process(self, image):
        """
        Process the input image
        :param image: the input image
        :return: the processed image
        """
        # Resize the image
        image = cv2.resize(image, (self.img_size, self.img_size))
        return image

    def detect_faces(self, image):
        """
        Detects faces in the input image and returns a list of bounding boxes
        corresponding to the present faces
        :param image: The input image
        :return: A list of bounding boxes
        """
        return []
