import tensorflow as tf
import numpy as np
import argparse
from inception_resnet_v1 import   inference



#input = tf.placeholder(tf.float32,shpae=(None,182,182  ,3),name='input')
#positive = tf.placeholder (tf.float32, shpae=(None,182,182  ,3), name='positive')
#negative = tf.placeholder (tf.float32, shpae=(None,182,182  ,3), name='negative')


def triplet_loss (anchor=tf.placeholder, positive=tf.placeholder, negative=tf.placeholder, alpha=0.2):
	"""Calculate the triplet loss according to the FaceNet paper

	Args:
	  anchor: the embeddings for the anchor images.
	  positive: the embeddings for the positive images.
	  negative: the embeddings for the negative images.
	  alpha:

	Returns:
	  the triplet loss according to the FaceNet paper as a float tensor.

	"""
	pos_dist = tf.reduce_sum (tf.square (tf.subtract (anchor, positive)), 1)
	neg_dist = tf.reduce_sum (tf.square (tf.subtract (anchor, negative)), 1)
	basic_loss = tf.add (tf.subtract (pos_dist, neg_dist), alpha)
	loss = tf.reduce_mean (tf.maximum (basic_loss, 0.0), 0)

	return loss


def total_loss (triple_loss):
	regularization_losses = tf.get_collection (tf.GraphKeys.REGULARIZATION_LOSSES)
	return tf.add_n ([ triple_loss ] + regularization_losses, name='total_loss')

def inception (input, dropout_keep_prob=0.8, is_training=True,
                        bottleneck_layer_size=128,weight_decay=0.0 ,reuse=None):

	net,end_points =  inference(input,dropout_keep_prob,is_training,bottleneck_layer_size,weight_decay,reuse)
	embedings  = tf.nn.l2_normalize(net, 1, 1e-10, name='embeddings')
	return embedings , end_points

'''def parse_arguments (argv) :
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_size', type=str, help='',default=(100,100))
	parser.add_argument ('--alpa', type =int , help='', default=0.2)
	return  parser.parse_args(argv)'''

def make_pso_neg( shape = (None,182,182,3)):
		positive = tf.placeholder(tf.float32,shape=shape,name='positive')
		negative = tf.placeholder(tf.float32,shape=shape,name='negative')
		return  positive,negative

def train () :
	pass