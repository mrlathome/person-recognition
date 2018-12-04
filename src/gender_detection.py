from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
from keras.models import load_model


class Gender:
	def __init__(self, model= "pre_trained_gn/gender_detection.model"):
		self.model = load_model(model)
		self.classes = ['man', 'woman']

	def f_detector(self, frame, bbox):
		if frame is None:
			print("Could not read input image")
		gender = list()
		info = {}
		if np.shape(bbox)[0] is 0 :
			info = {"person": 0, "man": 0, "woman": 0}
			return info
		else:
			for idx, f in enumerate(bbox):
				(startx, starty) = f[0], f[1]
				(endx, endy) = f[2], f[3],
				conf = self.model.predict(
					self.crop(frame, startx, starty, endx, endy))[0]
				idx = np.argmax(conf)
				label = self.classes[idx]
				gender.append(label)
				info = {"person": np.shape(bbox)[0], "man": gender.count("man"), "woman": gender.count("woman")}
			return info, gender

	def m_detector(self, img):
		if img is None:
			print("Could not read input image")
		face_crop = cv2.resize(img, (96, 96))
		face_crop = face_crop.astype("float") / 255.0
		face_crop = img_to_array(face_crop)
		face_crop = np.expand_dims(face_crop, axis=0)
		conf = self.model.predict(face_crop)[0]
		idx = np.argmax(conf)
		label = self.classes[idx]
		return label


	def crop (self, frame, startx, starty, endx, endy):
		face_crop = np.copy(frame[int(np.rint(starty)):int(np.rint(endy)), int(np.rint(startx)):int(np.rint(endx))])
		face_crop = cv2.resize(face_crop, (96, 96))
		face_crop = face_crop.astype("float") / 255.0
		face_crop = img_to_array(face_crop)
		face_crop = np.expand_dims(face_crop, axis=0)
		return face_crop
