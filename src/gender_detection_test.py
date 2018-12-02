import unittest
import cv2
from gender_detection import Gender
from align import detect_face
import tensorflow as tf
import numpy as np

class GenderTestCase(unittest.TestCase):
	def test_evaluate(self):
		self.gn = Gender()
		self.img = cv2.imread("/Users/amir/PycharmProjects/person_recognition/dataset/train/0004.0001.jpg")
		self.model = self.gn.model
		gpu_memory_fraction = 1.0
		minsize = 50
		threshold = [0.6, 0.6, 0.7]
		factor = 0.709
		print('Creating networks and loading parameters')
		with tf.Graph().as_default():
			gpu_options = tf.GPUOptions(
				per_process_gpu_memory_fraction=gpu_memory_fraction)
			sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
			with sess.as_default():
				pnet, rnet, onet = detect_face.create_mtcnn(
					sess, None)
		bounding_boxes, acc = detect_face.detect_face(
			self.img, minsize, pnet,
			rnet, onet, threshold, factor)
		print self.img
		(startx, starty) = int (bounding_boxes[0][0]), int (bounding_boxes[0][1])
		(endx, endy) = int (bounding_boxes[0][2]), int(bounding_boxes[0][3])
		print startx , starty , endx, endy
		img= np.copy(self.img[starty:endy,startx:endx])
		print img
		label= self.gn.m_detector(img)
		print("label : {}".format( label))
		condition = label == "woman"
		self.assertEqual(condition, True)
		return label


if __name__ == '__main__':
	unittest.main()
