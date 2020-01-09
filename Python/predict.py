import tensorflow as tf 
tf.compat.v1.disable_eager_execution()
from utilities import predict, read_data, normalize_and_reshape
from net.net import CNN
import numpy as np 
import pandas as pd
import argparse
import pdb

def parse_args():
	parser = argparse.ArgumentParser(description='prediction')
	parser.add_argument('--test_set', dest='test_set',
					  help='provide the directory of .mat file for testing',
					  default='data/mnist-demo.t.mat', type=str)
	parser.add_argument('--model', dest='model_file',
					  help='provide file storing network parameters, i.e. ./dir/model.ckpt',
					  default='./saved_model/model.ckpt', type=str)
	parser.add_argument('--bsize', dest='bsize',
					  help='batch size',
					  default=1024, type=int)
	parser.add_argument('--loss', dest='loss', 
					  help='which loss function to use: MSELoss or CrossEntropy',
					  default='MSELoss', type=str)
	parser.add_argument('--dim', dest='dim', nargs='+', help='input dimension of data,'+\
						'shape must be:  height width num_channels',
					  default=[32, 32, 3], type=int)
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()

	sess_config = tf.compat.v1.ConfigProto()
	sess_config.gpu_options.allow_growth = True

	with tf.compat.v1.Session(config=sess_config) as sess:
		graph_address = args.model_file + '.meta'
		imported_graph = tf.compat.v1.train.import_meta_graph(graph_address)
		imported_graph.restore(sess, args.model_file)
		mean_param = [v for v in tf.compat.v1.global_variables() if 'mean_tr:0' in v.name][0]
		label_enum_var = [v for v in tf.compat.v1.global_variables() if 'label_enum:0' in v.name][0]
		
		sess.run(tf.compat.v1.variables_initializer([mean_param, label_enum_var]))
		mean_tr = sess.run(mean_param)
		label_enum = sess.run(label_enum_var)

		test_batch, num_cls, _ = read_data(args.test_set, label_enum=label_enum)
		test_batch[0], _ = normalize_and_reshape(test_batch[0], dim=args.dim, mean_tr=mean_tr)

		x = tf.compat.v1.get_default_graph().get_tensor_by_name('main_params/input_of_net:0')
		y = tf.compat.v1.get_default_graph().get_tensor_by_name('main_params/labels:0')
		outputs = tf.compat.v1.get_default_graph().get_tensor_by_name('output_of_net:0')

		if args.loss == 'MSELoss':
			loss = tf.reduce_sum(input_tensor=tf.pow(outputs-y, 2))
		else:
			loss = tf.reduce_sum(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=tf.stop_gradient(y)))
		
		network = (x, y, loss, outputs)

		avg_loss, avg_acc, results = predict(sess, network, test_batch, args.bsize)

		# convert results back to the original labels
		inverse_map = dict(zip(np.arange(num_cls), label_enum))
		results = np.expand_dims(results, axis=1)
		results = np.apply_along_axis(lambda x: inverse_map[x[0]], axis=1, arr=results)
	
	print('In test phase, average loss: {:.3f} | average accuracy: {:.3f}%'.\
		format(avg_loss, avg_acc*100))
