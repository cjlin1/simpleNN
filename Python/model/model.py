import tensorflow as tf
import math
import pdb
from tensorflow.python.client import device_lib

def CNN_3layers(x_image, reuse=False):
	_NUM_CLASSES = 10
	with tf.variable_scope('conv1', reuse=reuse) as scope:
		conv = tf.keras.layers.Conv2D(
			filters=32,
			kernel_size=[5, 5],
			padding='SAME',
			activation=tf.nn.relu
		)(x_image)
		pool = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='SAME')(conv)
		# N x 16 x 16 x 32

	with tf.variable_scope('conv2', reuse=reuse) as scope:
		conv = tf.keras.layers.Conv2D(
			filters=64,
			kernel_size=[3, 3],
			padding='SAME',
			activation=tf.nn.relu
		)(pool)
		pool = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='SAME')(conv)
		# N x 8 x 8 x 64
		
	with tf.variable_scope('conv3', reuse=reuse) as scope:
		conv = tf.keras.layers.Conv2D(
			filters=64,
			kernel_size=[3, 3],
			padding='SAME',
			activation=tf.nn.relu
		)(pool)
		pool = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='SAME')(conv)
		# N x 4 x 4 x 64

	with tf.variable_scope('fully_connected', reuse=reuse) as scope:
		flat = tf.reshape(pool, [-1, 4 * 4 * 64])
		outputs = tf.keras.layers.Dense(units=_NUM_CLASSES, name=scope.name)(flat)

	return outputs

def CNN_6layers(x_image, reuse=False):
	_NUM_CLASSES = 10
	with tf.variable_scope('conv1', reuse=reuse) as scope:
		conv = tf.keras.layers.Conv2D(
			filters=32,
			kernel_size=[5, 5],
			padding='SAME',
			activation=tf.nn.relu
		)(x_image)
		conv = tf.keras.layers.Conv2D(
			filters=32,
			kernel_size=[3, 3],
			padding='SAME',
			activation=tf.nn.relu
		)(conv)
		pool = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='SAME')(conv)
		# N x 16 x 16 x 32

	with tf.variable_scope('conv2', reuse=reuse) as scope:
		conv = tf.keras.layers.Conv2D(
			filters=64,
			kernel_size=[3, 3],
			padding='SAME',
			activation=tf.nn.relu
		)(pool)
		conv = tf.keras.layers.Conv2D(
			filters=64,
			kernel_size=[3, 3],
			padding='SAME',
			activation=tf.nn.relu
		)(conv)
		pool = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='SAME')(conv)
		# N x 8 x 8 x 64

	with tf.variable_scope('conv3', reuse=reuse) as scope:
		conv = tf.keras.layers.Conv2D(
			filters=64,
			kernel_size=[3, 3],
			padding='SAME',
			activation=tf.nn.relu
		)(pool)
		conv = tf.keras.layers.Conv2D(
			filters=128,
			kernel_size=[3, 3],
			padding='SAME',
			activation=tf.nn.relu
		)(conv)
		pool = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='SAME')(conv)
		# pool = tf.layers.dropout(pool, rate=0.25, name=scope.name)
		# N x 4 x 4 x 128

	with tf.variable_scope('fully_connected', reuse=reuse) as scope:
		flat = tf.reshape(pool, [-1, 4 * 4 * 128])
		outputs = tf.keras.layers.Dense(units=_NUM_CLASSES, name=scope.name)(flat)

	return outputs

def CNN(config):
	_NUM_CLASSES = 10
	_IMAGE_SIZE = 32
	_IMAGE_CHANNELS = 3
	with tf.name_scope('main_params'):
		x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='Input')
		y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')

	net = CNN_3layers if config.net == 'CNN_3layers' else CNN_6layers
	outputs = net(x)

	return (x, y, outputs)
