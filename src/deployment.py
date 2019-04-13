#!/usr/bin/env python3

"""
Handles ros operations necessary for using the package.
"""
import os
import roslib
import rospkg
import rospy
from person_recognition.srv import *
from std_msgs.msg import String
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
        rospy.Service('person_recognition/detect', DetectCrowd, self.handle_detect_crowd)
        accuracy = self.execution.evaluate()
        rospy.loginfo('Evaluation result: {}'.format(accuracy))
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

    def handle_detect_crowd(self, req):
        dict= self.execution.gendder_classifer()
        rospy.loginfo('gender : {}'.format(dict))
        persons = dict_to_json_str(dict)
        objects_message = String(data=persons)
        return DetectCrowdResponse(crowd=objects_message)

    def idle(self):
        try:
            print("Ready to recognize persons.")
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down person recognition module")

    def dict_to_json_str(self, data):
        json_data = dict()
        for key, value in data.iteritems():
            if isinstance(value, list):  # for lists
                value = [json.dumps(item) if isinstance(item, dict)
                         else item for item in value]
            if isinstance(value, dict):  # for nested lists
                value = json.dumps(value)
            if isinstance(key, int):  # if key is integer: > to string
                key = str(key)
            if type(value).__module__ == 'numpy':  # if value is numpy.*: > to python list
                value = value.tolist()
            json_data[key] = value
        return json.dumps(json_data, indent=4)


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
