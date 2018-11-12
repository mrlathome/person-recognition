#!/usr/bin/env python3

"""
Handles ros operations necessary for using the package.
"""
import os

import roslib
import rospkg
import rospy
from person_recognition.srv import *

from execution import Execution

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

package_name = 'person_recognition'
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
        # eval_result = self.execution.evaluate()
        # print('Evaluation result:', eval_result)
        rospy.Service('person_recognition/add', AddPerson, self.handle_add_person)
        rospy.Service('person_recognition/delete', DeletePerson, self.handle_delete_person)
        self.execution.test()

    def handle_add_person(self, req):
        name = req.name.data
        rospy.loginfo('Adding name: {}'.format(name))
        for i in range(4):
            if self.execution.add_person(name):
                break
        return AddPersonResponse()

    def handle_delete_person(self, req):
        name = req.name.data
        rospy.loginfo('Deleting name: {}'.format(name))
        for i in range(4):
            if self.execution.delete_person(name):
                break
        return DeletePersonResponse()

    def idle(self):
        try:
            print("Ready to recognize persons.")
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down person recognition module")


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
