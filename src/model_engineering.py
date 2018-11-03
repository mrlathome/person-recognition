"""
Loading models and running inferences
"""
import os
import re

import numpy as np
import tensorflow as tf
from sklearn import neighbors
from tensorflow.python.platform import gfile


class ModelEngineering:
    def __init__(self, pkg_dir):
        self.pkg_dir = pkg_dir
        self.frozen_graph_path = os.path.join(pkg_dir, 'InceptionResNetV1-VGGFace2', '20180402-114759.pb')
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        self.imgs_ph = None
        self.phase_train_ph = None
        self.embs_ph = None
        self.emb_size_ph = None
        self.initialized = False
        # we create an instance of Neighbours Classifier and fit the data.
        self.n_neighbors = 2
        # weight function used in prediction. Possible values: 'uniform', 'distance', [callable]
        self.weights = 'distance'
        self.clf = neighbors.KNeighborsClassifier(self.n_neighbors, algorithm='ball_tree', weights=self.weights)

    def initialize(self):
        """
        Call load_model method and get input/output tensors
        :return: True, if everything goes well
        """
        self.imgs_ph, self.phase_train_ph, self.embs_ph, self.emb_size_ph = self.load_model(self.frozen_graph_path)
        return True

    def load_model(self, model, input_map=None):
        """
        Load a (frozen) Tensorflow model into memory.
        :param model: Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file
        :param input_map: The input map
        :return: The place holders for input dataset, phase train, embeddings, and the embedding size
        """
        with self.graph.as_default():
            # Check if the model is a model directory (containing a metagraph and a checkpoint file)
            #  or if it is a protobuf file with a frozen graph
            model_exp = os.path.expanduser(model)
            if os.path.isfile(model_exp):
                print('Model filename: %s' % model_exp)
                with gfile.FastGFile(model_exp, 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, input_map=input_map, name='')
            else:
                print('Model directory: %s' % model_exp)
                meta_file, ckpt_file = self.get_model_filenames(model_exp)

                print('Metagraph file: %s' % meta_file)
                print('Checkpoint file: %s' % ckpt_file)

                saver = tf.train.import_meta_graph(os.path.join(
                    model_exp, meta_file), input_map=input_map)
                saver.restore(self.session, os.path.join(model_exp, ckpt_file))

            # Get input and output tensors
            imgs_ph = self.graph.get_tensor_by_name("input:0")
            embs_ph = self.graph.get_tensor_by_name("embeddings:0")
            phase_train_ph = self.graph.get_tensor_by_name("phase_train:0")
            emb_size = embs_ph.get_shape()[1]

        return imgs_ph, phase_train_ph, embs_ph, emb_size

    @staticmethod
    def get_model_filenames(model_dir):
        """
        Get the model file names.
        :param model_dir: The directory in which the saved checkpoints of the model exists.
        :return: The meta file name and the checkpoint file name
        """
        files = os.listdir(model_dir)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files) == 0:
            raise ValueError(
                'No meta file found in the model directory (%s)' % model_dir)
        elif len(meta_files) > 1:
            raise ValueError(
                'There should not be more than one meta file in the model directory ({})'.format(model_dir))
        meta_file = meta_files[0]
        ckpt = tf.train.get_checkpoint_state(model_dir)
        ckpt_file = ''
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
            return meta_file, ckpt_file

        max_step = -1
        for f in files:
            step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
            if step_str is not None and len(step_str.groups()) >= 2:
                step = int(step_str.groups()[1])
                if step > max_step:
                    max_step = step
                    ckpt_file = step_str.groups()[0]
        return meta_file, ckpt_file

    def encode(self, images):
        """
        Run the forward pass to calculate embeddings.
        :param images: The input (4D) tensor
        :return: The 512-vector embeddings
        """
        if not self.initialized:
            self.initialized = self.initialize()

        feed_dict = {self.imgs_ph: images, self.phase_train_ph: False}
        emb_array = self.session.run(self.embs_ph, feed_dict=feed_dict)
        return emb_array

    def knn_fit(self, warehouse):
        """
        Fit the KNN classifier using the training data set
        :param warehouse:
        :return:
        """
        emb_array = np.array([])
        uid_array = np.array([])
        for sample in warehouse.get_samples():
            if emb_array.ndim == 1:
                emb_array = sample.embedding
            else:
                emb_array = np.vstack((emb_array, sample.embedding))
            uid_array = np.append(uid_array, sample.uid)
        self.clf.fit(emb_array, uid_array)

    def knn_classify(self, query):
        """
        Supervised KNN
        :param query: the subject embedding
        :return: the UID of the subject
        """
        proba = self.clf.predict_proba([query])[0]
        index = np.argmax(proba)
        uid = -1
        if proba[index] > 0.5:
            uid = index
        # print('proba[index]', proba[index])
        # print('detect_uid', uid)
        return uid

    def knn_eval(self, warehouse):
        """
        Evaluate the KNN classifier on a test data set
        :return: the accuracy
        """
        emb_array = np.array([])
        uid_array = np.array([])
        for sample in warehouse.get_samples():
            if emb_array.ndim == 1:
                emb_array = sample.embedding
            else:
                emb_array = np.vstack((emb_array, sample.embedding))
            uid_array = np.append(uid_array, sample.uid)
        accuracy = self.clf.score(emb_array, uid_array)
        return accuracy

    def __knn_classify(self, warehouse, query_emb):
        """
        Classify the query embedding
        :param warehouse: the training warehouse
        :param query_emb: the embedding to be classified
        :return: the UID corsresponding to the query
        """
        with tf.Session() as sess:
            new_sample = tf.reduce_mean(query_emb)
            distance = {}
            for samples in warehouse:
                embs = []
                for sample in samples:
                    embs.append(sample.embedding)
                cnt_class = sess.run(tf.reduce_mean(
                    tf.reduce_sum(np.asarray(embs), 1)), )
                dis = sess.run(tf.abs(
                    tf.add(cnt_class, tf.negative(new_sample))))
                distance.update({"{}".format(sample.uid): dis})

        return min(distance.items(), key=lambda x: x[1])[0]
