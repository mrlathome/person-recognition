"""
Handles ros operations necessary for using the package.
"""

import roslib
import rospkg
import numpy as np
from execution import Execution
import copy
import cv2

package_name = 'ros-person-recognition'
roslib.load_manifest(package_name)
rp = rospkg.RosPack()
pkg_dir = rp.get_path(package_name)


class Deployment:
    """
    Performs the functions for serving the customer.
    """

    def __init__(self, pkg_dir):
        self.pkg_dir = pkg_dir
        self.execution = Execution(self.pkg_dir)

    def visualize(self, image, bboxes, uids):
        """
        Draws bounding boxes corresponding to faces on the input image
        with UIDs at the top-left corner of each rectangle
        :param image: The input image containing faces
        :param bboxes: The bounding boxes
        :param uids: The UIDs
        :return: The image overlaid with the detection and recognition info
        """
        for bbox, uid in zip(bboxes, uids):
            _image = copy.deepcopy(image)
            xmin, xmax, ymin, ymax = bbox
            color_b, color_g, color_r = np.random.random_sample((3,)) * 255
            thickness = 3
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (color_b, color_g, color_r), thickness)
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            cv2.putText(_image, id, (xmin, ymin), font_face, font_scale,
                        (color_b, color_g, color_r), thickness, cv2.LINE_AA)
        return _image


def main():
    """
    Runs the program.
    :return:
    """
    rospy.init_node(package_name)

    deployment = Deployment(pkg_dir)
    deployment.run()


if __name__ == '__main__':
    main()
