import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math
import pdb
from tensorflow.python.client import device_lib
import numpy as np
from net.vgg import *

def CNN_4layers(x_image, num_cls, reuse=False):
	_NUM_CLASSES = num_cls
	with tf.variable_scope('conv1', reuse=reuse) as scope:
		conv = tf.keras.layers.Conv2D(
			filters=32,
			kernel_size=[5, 5],
			padding='SAME',
			activation=tf.nn.relu
		)(x_image)
		pool = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='valid')(conv)
		# N x 16 x 16 x 32

	with tf.variable_scope('conv2', reuse=reuse) as scope:
		conv = tf.keras.layers.Conv2D(
			filters=64,
			kernel_size=[3, 3],
			padding='SAME',
			activation=tf.nn.relu
		)(pool)
		pool = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='valid')(conv)
		# N x 8 x 8 x 64
		
	with tf.variable_scope('conv3', reuse=reuse) as scope:
		conv = tf.keras.layers.Conv2D(
			filters=64,
			kernel_size=[3, 3],
			padding='SAME',
			activation=tf.nn.relu
		)(pool)
		pool = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='valid')(conv)
		# N x 4 x 4 x 64

	with tf.variable_scope('fully_connected', reuse=reuse) as scope:
		dim =np.prod(pool.shape[1:])
		flat = tf.reshape(pool, [-1, dim])
		outputs = tf.keras.layers.Dense(units=_NUM_CLASSES, name=scope.name)(flat)

	return outputs

def CNN_7layers(x_image, num_cls, reuse=False):
	_NUM_CLASSES = num_cls
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
		pool = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='valid')(conv)
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
		pool = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='valid')(conv)
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
		pool = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='valid')(conv)
		# pool = tf.layers.dropout(pool, rate=0.25, name=scope.name)
		# N x 4 x 4 x 128

	with tf.variable_scope('fully_connected', reuse=reuse) as scope:
		dim = np.prod(pool.shape[1:])
		flat = tf.reshape(pool, [-1, dim])
		outputs = tf.keras.layers.Dense(units=_NUM_CLASSES, name=scope.name)(flat)

	return outputs

def CNN(net, num_cls, dim):

	_NUM_CLASSES = num_cls
	_IMAGE_HEIGHT, _IMAGE_WIDTH, _IMAGE_CHANNELS = dim

	with tf.name_scope('main_params'):
		x = tf.placeholder(tf.float32, shape=[None, _IMAGE_HEIGHT, _IMAGE_WIDTH, _IMAGE_CHANNELS], name='input_of_net')
		y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='labels')

	# call CNN structure according to string net
	outputs = globals()[net](x, _NUM_CLASSES)
	outputs = tf.identity(outputs, name='output_of_net')

	return (x, y, outputs)
