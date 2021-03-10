import math
import os

import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.python import _pywrap_stat_summarizer


class ConfigClass(object):
	def __init__(self, args, num_data, num_cls):
		super(ConfigClass, self).__init__()
		self.args = args
		self.iter_max = args.iter_max
		
		# Different notations of regularization term:
		# In SGD, weight decay:
		# 	weight_decay <- lr/(C*num_of_training_samples)
		# In Newton method:
		# 	C <- C * num_of_training_samples

		self.seed = args.seed

		if self.seed is None:
			print('You choose not to specify a random seed.'+\
				'A different result is produced after each run.')
		elif isinstance(self.seed, int) and self.seed >= 0:
			print('You specify random seed {}.'.format(self.seed))
		else:
			raise ValueError('Only accept None type or nonnegative integers for'+\
					' random seed argument!')

		self.train_set = args.train_set
		self.val_set = args.val_set
		self.num_cls = num_cls
		self.dim = args.dim

		self.num_data = num_data
		self.GNsize = min(args.GNsize, self.num_data)
		self.C = args.C * self.num_data
		self.net = args.net

		self.xi = 0.1
		self.CGmax = args.CGmax
		self._lambda = args._lambda
		self.drop = args.drop
		self.boost = args.boost
		self.eta = args.eta
		self.lr = args.lr
		self.lr_decay = args.lr_decay

		self.bsize = args.bsize
		if args.momentum < 0:
			raise ValueError('Momentum needs to be larger than 0!')
		self.momentum = args.momentum

		self.loss = args.loss
		if self.loss not in ('MSELoss', 'CrossEntropy'):
			raise ValueError('Unrecognized loss type!')
		self.optim = args.optim
		if self.optim not in ('SGD', 'NewtonCG', 'Adam'):
			raise ValueError('Only support SGD, Adam & NewtonCG optimizer!')
		
		self.log_file = args.log_file
		self.model_file = args.model_file
		self.screen_log_only = args.screen_log_only

		if self.screen_log_only:
			print('You choose not to store running log. Only store model to {}'.format(self.log_file))
		else:
			print('Saving log to: {}'.format(self.log_file))
			dir_name, _ = os.path.split(self.log_file)
			if not os.path.isdir(dir_name):
				os.makedirs(dir_name, exist_ok=True)

		dir_name, _ = os.path.split(self.model_file)
		if not os.path.isdir(dir_name):
			os.makedirs(dir_name, exist_ok=True)
		
		self.elapsed_time = 0.0

def read_data(filename, dim, label_enum=None):
	"""
	args:
		filename: the path where .mat files are stored
		label_enum (default None): the list that stores the original labels. 
			If label_enum is None, the function will generate a new list which stores the 
			original labels in a sequence, and map original labels to [0, 1, ... number_of_classes-1]. 
			If label_enum is a list, the function will use it to convert 
			original labels to [0, 1,..., number_of_classes-1].
	"""

	mat_contents = sio.loadmat(filename)
	images, labels = mat_contents['Z'], mat_contents['y']
	
	labels = labels.reshape(-1)
	images = images.reshape(images.shape[0], -1)

	_IMAGE_HEIGHT, _IMAGE_WIDTH, _IMAGE_CHANNELS = dim
	zero_to_append = np.zeros((images.shape[0],
			_IMAGE_CHANNELS*_IMAGE_HEIGHT*_IMAGE_WIDTH-np.prod(images.shape[1:])))
	images = np.append(images, zero_to_append, axis=1)

	# check data validity
	if label_enum is None:
		label_enum, labels = np.unique(labels, return_inverse=True)
		num_cls = labels.max() + 1

		if len(label_enum) != num_cls:
			raise ValueError('The number of classes is not equal to the number of\
							labels in dataset. Please verify them.')
	else:
		num_cls = len(label_enum)
		forward_map = dict(zip(label_enum, np.arange(num_cls)))
		labels = np.expand_dims(labels, axis=1)
		labels = np.apply_along_axis(lambda x:forward_map[x[0]], axis=1, arr=labels)
		

	# convert groundtruth to one-hot encoding
	labels = np.eye(num_cls)[labels]
	labels = labels.astype('float32')

	return [images, labels], num_cls, label_enum

def normalize_and_reshape(images, dim, mean_tr=None):
	_IMAGE_HEIGHT, _IMAGE_WIDTH, _IMAGE_CHANNELS = dim
	images_shape = [images.shape[0], _IMAGE_CHANNELS, _IMAGE_HEIGHT, _IMAGE_WIDTH]

	# images normalization and zero centering
	images = images.reshape(images_shape[0], -1)

	images = images/255.0

	if mean_tr is None:
		print('No mean of data provided! Normalize images by their own mean.')
		# if no mean_tr is provided, we calculate it according to the current data
		mean_tr = images.mean(axis=0) 
	else:
		print('Normalzie images according to the provided mean.')
		if np.prod(mean_tr.shape) != np.prod(dim):
			raise ValueError('Dimension of provided mean does not agree with the data! Please verify them!')

	images = images - mean_tr

	images = images.reshape(images_shape)
	# Tensorflow accepts data shape: B x H x W x C
	images = np.transpose(images, (0, 2, 3, 1))
	return images, mean_tr


def predict(sess, network, test_batch, bsize):
	x, y, loss, outputs = network

	test_inputs, test_labels = test_batch
	batch_size = bsize

	num_data = test_labels.shape[0]
	num_batches = math.ceil(num_data/batch_size)

	results = np.zeros(shape=num_data, dtype=np.int)
	infer_loss = 0.0

	for i in range(num_batches):
		batch_idx = np.arange(i*batch_size, min((i+1)*batch_size, num_data))

		batch_input = test_inputs[batch_idx]
		batch_labels = test_labels[batch_idx]

		net_outputs, _loss = sess.run(
			[outputs, loss], feed_dict={x: batch_input, y: batch_labels}
			)
		
		results[batch_idx] = np.argmax(net_outputs, axis=1)
		# note that _loss was summed over batches
		infer_loss = infer_loss + _loss

	avg_acc = (np.argmax(test_labels, axis=1) == results).mean()
	avg_loss = infer_loss/num_data
	
	return avg_loss, avg_acc, results


class Profiler:
	def __init__(self, is_enabled=False):
		self._is_enabled = is_enabled
		self.run_metadata = None
		self._summarizer = _pywrap_stat_summarizer.StatSummarizer()

		if self._is_enabled:
			self.run_options = tf.compat.v1.RunOptions(
				trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
		else:
			self.run_options = None

	def add_stat(self, run_metadata):
		self._summarizer.ProcessStepStatsStr(
			run_metadata.step_stats.SerializeToString())

	def __enter__(self):
		if self._is_enabled:
			if self.run_metadata is not None:
				raise RuntimeError('Recursively called')
			self.run_metadata = tf.compat.v1.RunMetadata()
		return self

	def __exit__(self, *args, **kwargs):
		if self._is_enabled:
			self.add_stat(self.run_metadata)
			self.run_metadata = None

	def summary(self):
		return self._summarizer.GetOutputString()
