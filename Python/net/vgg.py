"""
Codes are modifeid from PyTorch and Tensorflow Versions of VGG: 
https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py, and
https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py
"""

import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras as keras
tf.disable_v2_behavior()
import numpy as np

__all__ = ['VGG11', 'VGG13', 'VGG16','VGG19']

def make_layers(layer_spec, input_shape, output_shape):
	layers = [
		keras.layers.Conv2D(
			filters=layer_spec[0],
			kernel_size=[3, 3],
			padding='SAME',
			activation=tf.nn.relu,
			input_shape=input_shape,
		),
	]
	for v in layer_spec[1:]:
		if v == 'M':
			layers.append(keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='valid'))
		else:
			layers.append(keras.layers.Conv2D(
				filters=v,
				kernel_size=[3, 3],
				padding='SAME',
				activation=tf.nn.relu
			))

	layers.append(keras.layers.Flatten())
	layers.append(keras.layers.Dense(units=4096, activation='relu'))
	layers.append(keras.layers.Dense(units=4096, activation='relu'))
	layers.append(keras.layers.Dense(units=output_shape))

	return layers

layer_spec = {
	'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
		  512, 512, 512, 512, 'M'],
}

def VGG11(input_shape, output_shape):
	layers = make_layers(layer_spec['A'], input_shape, output_shape)
	return keras.models.Sequential(layers)

def VGG13(input_shape, output_shape):
	layers = make_layers(layer_spec['B'], input_shape, output_shape)
	return keras.models.Sequential(layers)

def VGG16(input_shape, output_shape):
	layers = make_layers(layer_spec['D'], input_shape, output_shape)
	return keras.models.Sequential(layers)

def VGG19(input_shape, output_shape):
	layers = make_layers(layer_spec['E'], input_shape, output_shape)
	return keras.models.Sequential(layers)

	