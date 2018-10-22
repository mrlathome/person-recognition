"""
Loading models and running inferences
"""

import os
import re
import tensorflow as tf
from tensorflow.python.platform import gfile


class ModelEngineering:
    def __init__(self):
        self.img_w = 84
        self.img_h = 84
        self.embedding_size = 128
        self.path_frozen_graph = 'InceptionResNetV1-VGGFace2'
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        self.load_model(self.path_frozen_graph)
        self.imgs_ph, self.phase_train_ph, self.embs_ph, self.emb_size_ph = \
            self.get_in_out_tensors()

    @staticmethod
    def get_in_out_tensors():
        """
        Get input and output tensors.
        :return:
        """
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        return images_placeholder, phase_train_placeholder, embeddings, embedding_size

    def load_model(self, model, input_map=None):
        """
        Load a (frozen) Tensorflow model into memory.
        :param model: Could be either a directory containing the meta_file and ckpt_file
        or a model protobuf (.pb) file
        :param input_map:
        :return:
        """
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

            saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file),
                                               input_map=input_map)
            saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

    @staticmethod
    def get_model_filenames(model_dir):
        """
        Get model file names
        :param model_dir:
        :return:
        """
        files = os.listdir(model_dir)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files) == 0:
            raise ValueError('No meta file found in the model directory (%s)' % model_dir)
        elif len(meta_files) > 1:
            raise ValueError(
                'There should not be more than one meta file in the model directory ({})'.format(
                    model_dir))
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
        Run forward pass to calculate embeddings
        :param images: The input images tensor
        :return: The 128-vector encoding
        """
        feed_dict = {self.imgs_ph: images, self.phase_train_ph: False}
        emb_array = self.session.run(self.embs_ph, feed_dict=feed_dict)
        return emb_array
