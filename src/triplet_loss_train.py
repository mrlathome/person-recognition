import tensorflow as tf

from inception_resnet_v1 import inference


class InceptionTripletLoss:

	def __init__ (self):
		self.inputs_shape = (None, 182, 182, 3)
		# A 4-D tensor of size [batch_size, height, width, 3].
		self.inputs_ph = tf.placeholder (tf.float32, shape=self.inputs_shape, name='positive')
		self.positives_ph = tf.placeholder (tf.float32, shape=self.inputs_shape, name='positive')
		self.negatives_ph = tf.placeholder (tf.float32, shape=self.inputs_shape, name='negative')

		self.embeddings = self.build_model ()
		self.loss = self.triplet_loss ()
		self.adam_opt = self.optimizer ()

	def triplet_loss (self, alpha=0.2):
		"""
		Calculates the triplet loss according to the FaceNet paper.
		:param anchors: The embeddings of anchor samples.
		:param positives: The place holder for embeddings of positive samples.
		:param negatives: The place holder for embeddings of negative samples.
		:param alpha: The margin
		:return: The value of triplet loss
		"""

		pos_dist = tf.reduce_sum (tf.square (tf.subtract (self.inputs_ph, self.positives_ph)), 1)
		neg_dist = tf.reduce_sum (tf.square (tf.subtract (self.inputs_ph, self.negatives_ph)), 1)
		basic_loss = tf.add (tf.subtract (pos_dist, neg_dist), alpha)
		tri_loss = tf.reduce_mean (tf.maximum (basic_loss, 0.0), 0)

		regularization_losses = tf.get_collection (tf.GraphKeys.REGULARIZATION_LOSSES)
		total_loss = tf.add_n ([ tri_loss ] + regularization_losses, name='total_loss')

		return total_loss

	def build_model (self, dropout_keep_prob=0.8, is_training=True, bottleneck_layer_size=128,
	                 weight_decay=0.0):
		"""
		Builds the inception model.
		:param dropout_keep_prob: float, the fraction to keep before final layer
		:param is_training: Whether is training or not
		:param bottleneck_layer_size: The size of the logits outputs of the model
		:param weight_decay: This penalizes large weights
		:return: The normalized logits outputs of the model
		"""
		net, end_points = inference (self.inputs_ph, dropout_keep_prob, is_training,
		                             bottleneck_layer_size, weight_decay)
		embeddings = tf.nn.l2_normalize (net, 1, 1e-10, name='embeddings')
		return embeddings

	def optimizer (self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08):
		return tf.train.AdamOptimizer (learning_rate, beta1, beta2, epsilon)


def add_loss_summmary (self, log_dir=os.path.join (os.getcwd (), 'log'), tag='loss', value=0.00):
	if not os.path.exists (log_dir):
		os.mkdir (log_dir)
	s_writer = tf.summary.FileWriter (log_dir)
	s_scalar = tf.Summary ()
	s_scalar.value.add (tag=tag, simple_value=value)
	return s_writer, s_scalar
