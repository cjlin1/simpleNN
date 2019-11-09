import tensorflow as tf 
from utilities import predict, read_data
from net.net import CNN
import argparse
import pdb

def parse_args():
	parser = argparse.ArgumentParser(description='prediction')
	parser.add_argument('--test_set', dest='test_set',
					  help='provide the directory of .mat file for testing',
					  default='data/mnist-demo.t.mat', type=str)
	parser.add_argument('--model', dest='net_name',
					  help='provide file storing network parameters, i.e. ./dir/model.ckpt',
					  default='./saved_model/model.ckpt', type=str)
	parser.add_argument('--bsize', dest='bsize',
					  help='batch size',
					  default=256, type=int)
	parser.add_argument('--net', dest='net',
					  help='classifier type',
					  default='CNN_3layers', type=str)
	parser.add_argument('--loss', dest='loss', 
					  help='which loss function to use: MSELoss or CrossEntropy',
					  default='MSELoss', type=str)
	parser.add_argument('--dim', dest='dim', nargs='+', help='input dimension of data,'+\
						'shape must be:  height width num_channels',
					  default=[28, 28, 1], type=int)
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()
	test_batch, num_cls = read_data(args.test_set, dim=args.dim)
	if args.net not in ('CNN_3layers', 'CNN_6layers'):
		raise ValueError('Unrecognized training model')

	sess_config = tf.ConfigProto()
	sess_config.gpu_options.allow_growth = True

	with tf.Session(config=sess_config) as sess:
		graph_address = args.net_name + '.meta'
		imported_graph = tf.train.import_meta_graph(graph_address)
		imported_graph.restore(sess, args.net_name)

		x = tf.get_default_graph().get_tensor_by_name('main_params/input_of_net:0')
		y = tf.get_default_graph().get_tensor_by_name('main_params/labels:0')
		outputs = tf.get_default_graph().get_tensor_by_name('output_of_net:0')
		
		if args.loss == 'MSELoss':
			loss = tf.reduce_sum(tf.pow(outputs-y, 2))
		else:
			loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=y))
		
		network = (x, y, loss, outputs)

		avg_loss, avg_acc = predict(sess, network, test_batch, args.bsize)
	
	print('In test phase, average loss: {:.3f} | average accuracy: {:.3f}%'.\
		format(avg_loss, avg_acc*100))