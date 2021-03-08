import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math
import pdb
from tensorflow.python.client import device_lib
import numpy as np
import tensorflow.compat.v1.keras as keras
from net.vgg import *

def CNN_4layers(input_shape, output_shape):
	layers = [
		keras.layers.Conv2D(32, [5, 5], padding='same', activation=tf.nn.relu, input_shape=input_shape),
		keras.layers.MaxPool2D([2, 2], strides=2),
		keras.layers.Conv2D(64, [3, 3], padding='same', activation=tf.nn.relu),
		keras.layers.MaxPool2D([2, 2], strides=2),
		keras.layers.Conv2D(64, [3, 3], padding='same', activation=tf.nn.relu),
		keras.layers.MaxPool2D([2, 2], strides=2),
		keras.layers.Flatten(),
		keras.layers.Dense(output_shape),
	]
	return keras.models.Sequential(layers)

def CNN_7layers(input_shape, output_shape):
	layers = [
		keras.layers.Conv2D(32, [5, 5], padding='same', activation=tf.nn.relu, input_shape=input_shape),
		keras.layers.Conv2D(32, [3, 3], padding='same', activation=tf.nn.relu),
		keras.layers.MaxPool2D([2, 2], strides=2),
		keras.layers.Conv2D(64, [3, 3], padding='same', activation=tf.nn.relu),
		keras.layers.Conv2D(64, [3, 3], padding='same', activation=tf.nn.relu),
		keras.layers.MaxPool2D([2, 2], strides=2),
		keras.layers.Conv2D(64, [3, 3], padding='same', activation=tf.nn.relu),
		keras.layers.Conv2D(128, [3, 3], padding='same', activation=tf.nn.relu),
		keras.layers.MaxPool2D([2, 2], strides=2),
		keras.layers.Flatten(),
		keras.layers.Dense(output_shape),
	]
	return keras.models.Sequential(layers)

def CNN_model(net, input_shape, output_shape):
	return globals()[net](input_shape, output_shape)

def CNN(net, num_cls, dim):
	_NUM_CLASSES = num_cls
	_IMAGE_HEIGHT, _IMAGE_WIDTH, _IMAGE_CHANNELS = dim

	with tf.name_scope('main_params'):
		x = tf.placeholder(tf.float32, shape=[None, _IMAGE_HEIGHT, _IMAGE_WIDTH, _IMAGE_CHANNELS], name='input_of_net')
		y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='labels')

	outputs = CNN_model(net, dim, num_cls)(x)
	outputs = tf.identity(outputs, name='output_of_net')

	return (x, y, outputs)
