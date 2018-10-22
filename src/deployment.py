"""
Handles ros operations necessary for using the package.
"""

import roslib
import rospkg
import numpy as np
from execution import Execution

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

    def run(self):
        pass


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
