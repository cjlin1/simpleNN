"""
Codes are modifeid from PyTorch and Tensorflow Versions of VGG: 
https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py, and
https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np 
import pdb
from tensorflow.keras.applications.vgg16 import VGG16 as vgg16
from tensorflow.keras.applications.vgg19 import VGG19 as vgg19

__all__ = ['VGG11', 'VGG13', 'VGG16','VGG19']

def VGG(feature, num_cls):

	with tf.variable_scope('fully_connected') as scope:
		dim =np.prod(feature.shape[1:])
		x = tf.reshape(feature, [-1, dim])

		x = tf.keras.layers.Dense(units=4096, activation='relu', name=scope.name)(x)
		x = tf.keras.layers.Dense(units=4096, activation='relu', name=scope.name)(x)
		x = tf.keras.layers.Dense(units=num_cls, name=scope.name)(x)

	return x

def make_layers(x, cfg):
	for v in cfg:
		if v == 'M':
			x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='valid')(x)
		else:
			x = tf.keras.layers.Conv2D(
			filters=v,
			kernel_size=[3, 3],
			padding='SAME',
			activation=tf.nn.relu
			)(x)
	return x

cfg = {
	'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
		  512, 512, 512, 512, 'M'],
}

def VGG11(x_images, num_cls):
	feature = make_layers(x_images, cfg['A'])
	return VGG(feature, num_cls)

def VGG13(x_images, num_cls):
	feature = make_layers(x_images, cfg['B'])
	return VGG(feature, num_cls)

def VGG16(x_images, num_cls):
	feature = make_layers(x_images, cfg['D'])
	return VGG(feature, num_cls)

def VGG19(x_images, num_cls):
	feature = make_layers(x_images, cfg['E'])
	return VGG(feature, num_cls)

	