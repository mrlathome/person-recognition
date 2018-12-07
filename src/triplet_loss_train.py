import os
import tensorflow as tf
from inception_resnet_v1 import inference
import cv2
import numpy as np


class InceptionTripletLoss:

    def __init__(self):
        self.inputs_shape = (None, 182, 182, 3)
        self.inputs_ph, self.embeddings = self.build_model()
        self.positives_ph, self.negatives_ph, self.loss = self.triplet_loss()
        self.adam_opt = self.optimize()

    def triplet_loss(self, alpha=0.2):
        """
        Calculates the triplet loss according to the FaceNet paper.
        :param anchors: The embeddings of anchor samples.
        :param positives: The place holder for embeddings of positive samples.
        :param negatives: The place holder for embeddings of negative samples.
        :param alpha: The margin
        :return: The value of triplet loss
        """
        positives_ph = tf.placeholder(tf.float32, shape=self.inputs_shape, name='positive')
        negatives_ph = tf.placeholder(tf.float32, shape=self.inputs_shape, name='negative')
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(self.inputs_ph, positives_ph)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(self.inputs_ph, negatives_ph)), 1)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        tri_loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([tri_loss] + regularization_losses, name='total_loss')

        return positives_ph, negatives_ph, total_loss

    def build_model(self, dropout_keep_prob=0.8, is_training=True, bottleneck_layer_size=128,
                    weight_decay=0.0):
        """
        Builds the inception model.
        :param dropout_keep_prob: float, the fraction to keep before final layer
        :param is_training: Whether is training or not
        :param bottleneck_layer_size: The size of the logits outputs of the model
        :param weight_decay: This penalizes large weights
        :return: The normalized logits outputs of the model
        """
        # A 4-D tensor of size [batch_size, height, width, 3].
        inputs_ph = tf.placeholder(tf.float32, shape=self.inputs_shape, name='anchor')
        net, end_points = inference(inputs_ph, dropout_keep_prob, is_training,
                                    bottleneck_layer_size, weight_decay)
        embeddings = tf.nn.l2_normalize(net, 1, 1e-10, name='embeddings')
        return inputs_ph, embeddings

    def optimize(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08):
        """
        :param learning_rate: object
        :param beta1: object
        :param beta2: object
        :param epsilon: object
        :return: object
        """
        return tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(self.loss)

    @staticmethod
    def add_loss_summary(log_dir=os.path.join(os.getcwd(), 'log'), tags=None):
        """
        :param tags:
        :param log_dir: address directory for save loss summary
        :return:  s_scalar oject for set value loss and s_writer for write summary file
        """
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if tags is None:
            tags = ['loss', 'acurracy']
        summary_writer = tf.summary.FileWriter(log_dir)
        summary_scalar = tf.Summary()
        for tag in tags:
            if tag is 'loss':
                s_scalar.value.add(tag, simple_value=1.00)
            else:
                s_scalar.value.add(tag, simple_value=0.00)

        return summary_writer, summary_scalar

    def train(self, epoch=5, batch=10):
        summary_writer, summary_scalar = self.add_loss_summary(tags=['loss', 'acc'])
        with tf.Session() as sess:
            for eph in range(0, epoch):
                avg_loss = []
                avg_acc = []
                for bth in range(0, batch):
                    anchors = sess.run(self.embeddings, feed_dict={self.inputs_ph: anc_batch})
                    positives = sess.run(self.embeddings, feed_dict={self.positives_ph: pos_batch})
                    negatives = sess.run(self.embeddings, feed_dict={self.negatives_ph: neg_batch})
                    loss, output_adam = sess.run([self.loss, self.adam_opt], feed_dict={self.inputs_ph: anchors,
                                                                                        self.positives_ph: positives,
                                                                                        self.negatives_ph: negatives})
                    avg_loss.append(loss)
                summary_scalar.value[0].simple_value = np.mean(avg_loss)
                summary_writer.add_summary(summary_scalar, epoch)
