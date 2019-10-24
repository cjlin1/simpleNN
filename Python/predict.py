import tensorflow as tf 
from utilities import predict, read_data
from model.net import CNN
import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='prediction')
	parser.add_argument('--test_set', dest='test_set',
					  help='provide the directory of .mat file for testing',
					  default='data/cifar10.t.mat', type=str)
	parser.add_argument('--model', dest='net_name',
					  help='provide file storing network parameters, i.e. ./dir/best-model.ckpt',
					  default='./log_and_model/best-model.ckpt', type=str)
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
						'shape must be:  Height x Width x In_channel',
					  default=[32, 32, 3], type=int)
	parser.add_argument('--num_cls', dest='num_cls',
					  help='number of classes in the dataset',
					  default=10, type=int)	
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()
	test_batch = read_data(args.test_set, num_cls=args.num_cls, dim=args.dim)
	if args.net in ('CNN_3layers', 'CNN_6layers'):
		x, y, outputs = CNN(args)
		test_network = None
	else:
		raise ValueError('Unrecognized training model')
	
	if args.loss == 'MSELoss':
		loss = tf.reduce_sum(tf.pow(outputs-y, 2))
	else:
		loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=y))
	
	network = (x, y, loss, outputs)
	sess_config = tf.ConfigProto()
	sess_config.gpu_options.allow_growth = True

	with tf.Session(config=sess_config) as sess:
		saver = tf.train.Saver(tf.trainable_variables())
		saver.restore(sess, args.net_name)
		avg_loss, avg_acc = predict(sess, network, test_batch, args.bsize)
	
	print('In test phase, average loss: {:.3f} | average accuracy: {:.3f}%'.\
		format(avg_loss, avg_acc*100))