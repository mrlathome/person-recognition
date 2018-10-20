"""
Loading models and running inferences
"""

import tensorflow as tf
import numpy as np

from inception_resnet_v1 import inception_resnet_v1


class ModelEngineering:
    def __init__(self):
        self.img_w = 84
        self.img_h = 84
        self.model, self.inputs, self.outputs = self.build_model()
        self.path_frozen_graph = 'InceptionResNetV1-VGGFace2'
        self.graph = self.load_graph()
        self.session = tf.Session(graph=self.graph)

    def load_graph(self, model_file):
        """
        Load a frozen tensorflow model into memory.
        :param model_file: Path to the frozen graph
        :return: The pre-trained graph
        """

        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        return graph

    def get_handles(self, graph):
        """
        Get handles to input and output tensors.
        :return:
        """
        input_name = "import/"
        output_name = 'Bottleneck'
        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)

        return input_operation, output_operation

    def build_model(self):
        inputs = tf.placeholder(tf.float32, (None, self.img_w, self.img_h, 3), "inputs")
        inception = inception_resnet_v1(self.inputs)
        outputs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_network')
        with tf.variable_scope('InceptionResnetV1', reuse=True):
            outputs = tf.get_variable('Bottleneck')
            return inception, inputs, outputs

    def run(self, inputs):
        """
        Encode the input images
        :return:
        """
        with self.session(graph=self.graph) as sess:
            results = sess.run(self.output_operation.outputs[0], {
                self.input_operation.outputs[0]: t
            })
        results = np.squeeze(results)
        return results

    def get_tensor_image(self, image):
        """
        Read tensor from image file
        :param image:
        :return:
        """
        t = read_tensor_from_image_file(
            file_name,
            input_height=input_height,
            input_width=input_width,
            input_mean=input_mean,
            input_std=input_std)
        return t
