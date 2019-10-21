import numpy as np
import math
import scipy.io as sio
import pdb

def read_data(filename, num_cls, dim):

	mat_contents = sio.loadmat(filename)
	images, labels = mat_contents['Z'], mat_contents['y']
	
	labels = labels.reshape(-1)

	# check data validity
	nonrepeat = list(dict.fromkeys(labels))
	if len(nonrepeat) != num_cls:
		raise ValueError('The number of classes is not equal to the number of\
						labels in dataset. Please verify them.')
	
	labels = labels - min(nonrepeat)
	
	images_shape = [images.shape[0]]+dim

	# images normalization and zero centering
	images = images.reshape(images_shape[0], -1)

	max_value = images.max(axis=1).reshape(-1,1)
	min_value = images.min(axis=1).reshape(-1,1)
	
	images = (images - min_value) / (max_value - min_value)
	images = images - images.mean(axis=0)

	images = images.reshape(images_shape)
	# Tensorflow accepts data shape: B x H x W x C
	images = np.transpose(images, (0, 2, 3, 1))

	# convert groundtruth to one-hot encoding
	labels = np.eye(num_cls)[labels]
	labels = labels.astype('float32')
	
	return images, labels

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
	return avg_loss, avg_acc

if __name__ == '__main__':

	images, labels = read_data('data/cifar10.t.mat', 10, [3, 32, 32])
	pdb.set_trace()
