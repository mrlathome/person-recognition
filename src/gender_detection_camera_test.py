
import cv2
from  align import detect_face
import tensorflow as tf
from src.gender_detection import Gender

gnd = Gender()
gpu_memory_fraction = 1.0
minsize = 40
threshold = [0.6, 0.6, 0.7]
factor = 0.709
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

print('Creating networks and loading parameters')
with tf.Graph().as_default():
	gpu_options = tf.GPUOptions(
		per_process_gpu_memory_fraction=gpu_memory_fraction)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
	                                        log_device_placement=False))
	with sess.as_default():
		pnet, rnet, onet = detect_face.create_mtcnn(
			sess, None)
while True:
	ret_val, frame = webcam.read()
	bounding_boxes, acc = detect_face.detect_face(
		frame, minsize, pnet,
		rnet, onet, threshold, factor)
	if bounding_boxes.shape[0] is not 0:
		info, gender = gnd.f_detector(frame, bounding_boxes)
		print info
		for idx, f in enumerate(bounding_boxes):
			x1, y1 = f[0], f[1]
			x2, y2 = f[2], f[3]
			w = x2 - x1
			h = y2 - y1
			Y = y1 - 10 if y1 - 10 > 10 else y1 + 10
			cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w),
			                                          int(y1 + h)), (255, 0, 0), 2)
			cv2.putText(frame, gender[idx], (int(x1), int(Y)), cv2.FONT_HERSHEY_SIMPLEX,
			            0.7, (0, 255, 0), 2)
			cv2.putText(frame, "male: {}".format(info["man"]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
			            0.7, (0, 0, 255), 2)
			cv2.putText(frame, "female : {}".format(info["woman"]), (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
			            0.7, (255, 0, 0), 2)
			cv2.putText(frame, "people : {}".format(info["person"]), (10, 15), cv2.FONT_HERSHEY_SIMPLEX,
			            0.7, (255, 255, 0), 2)
	cv2.imshow('my webcam', frame)
	# cv2.imshow("crop imge",img[int(y1):int(y2), int(x1):int(x2)])
	if cv2.waitKey(1) == 27:
		break
cv2.destroyAllWindows()
