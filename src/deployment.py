#!/usr/bin/env python3

"""
Handles ros operations necessary for using the package.
"""

import roslib
import rospkg
import rospy

from execution import Execution

package_name = 'ros-person-recognition'
roslib.load_manifest(package_name)
rp = rospkg.RosPack()
pkg_dir = rp.get_path(package_name)


class Deployment:
    """
    Perform the actions to serve the customer.
    """

    def __init__(self, pkg_dir):
        self.pkg_dir = pkg_dir
        self.execution = Execution(self.pkg_dir)

    def run(self):
        result = self.execution.test()
        print('\n', 'Result:', result)


def main():
    """
    Run the program.
    :return:
    """
    rospy.init_node(package_name)

    deployment = Deployment(pkg_dir)
    deployment.run()


if __name__ == '__main__':
    main()
