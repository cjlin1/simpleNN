import tensorflow as tf 
from utilities import predict, read_data
from net.net import CNN
import argparse

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
					  default=256, type=int)
	parser.add_argument('--net', dest='net',
					  help='classifier type',
					  default='CNN_3layers', type=str)
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
	test_batch, num_cls = read_data(args.test_set, dim=args.dim)
	if args.net in ('CNN_3layers', 'CNN_6layers'):
		x, y, outputs = CNN(args.net, num_cls, args.dim)
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
		saver.restore(sess, args.model_file)
		avg_loss, avg_acc = predict(sess, network, test_batch, args.bsize)
	
	print('In test phase, average loss: {:.3f} | average accuracy: {:.3f}%'.\
		format(avg_loss, avg_acc*100))
