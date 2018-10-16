"""
Loading models and running inferences
"""

import tensorflow as tf


class ModelEngineering():
    def __init__(self):
        self.inception = None
        self.session = tf.Session()
        self.build_inception()
        self.load_inception_wieghts()

    def load_inception_weights(self):
        pass

    def build_inception(self):
        pass
