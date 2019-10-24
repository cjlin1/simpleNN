import pdb
import numpy as np
import tensorflow as tf
import time
import math
import argparse

from model.net import CNN
from newton_cg import newton_cg, Config
from utilities import read_data, predict

def parse_args():
	parser = argparse.ArgumentParser(description='Newton method on DNN')
	parser.add_argument('--C', dest='C',
					  help='regularization term, or so-called weight decay where'+\
					  		'weight_decay = lr/(C*num_of_samples) in this implementation' ,
					  default=math.inf, type=float)

	# Newton method arguments
	parser.add_argument('--s', dest='sample',
					  help='number of samples for estimating Gauss-Newton matrix',
					  default=4096, type=int)
	parser.add_argument('--iter_max', dest='iter_max',
					  help='the maximal number of Newton iterations',
					  default=100, type=int)
	parser.add_argument('--xi', dest='xi',
					  help='the tolerance in the relative stopping condition for CG',
					  default=0.1, type=float)
	parser.add_argument('--drop', dest='drop',
					  help='the drop constants for the LM method',
					  default=2/3, type=float)
	parser.add_argument('--boost', dest='boost',
					  help='the boost constants for the LM method',
					  default=3/2, type=float)
	parser.add_argument('--eta', dest='eta',
					  help='the parameter for the line search stopping condition',
					  default=0.0001, type=float)
	parser.add_argument('--CGmax', dest='CGmax',
					  help='the maximal number of CG iterations',
					  default=250, type=int)
	parser.add_argument('--lambda', dest='_lambda',
					  help='the initial lambda for the LM method',
					  default=1, type=float)

	# SGD arguments
	parser.add_argument('--epoch_max', dest='epoch',
					  help='number of training epoch',
					  default=500, type=int)
	parser.add_argument('--lr', dest='lr',
					  help='learning rate',
					  default=0.01, type=float)
	parser.add_argument('--decay', dest='lr_decay',
					  help='learning rate decay over each lr_decay epochs',
					  default=500, type=int)
	parser.add_argument('--momentum', dest='momentum',
					  help='momentum of learning',
					  default=0, type=float)

	# Model training arguments
	parser.add_argument('--bsize', dest='bsize',
					  help='batch size to evaluate stochastic gradient, Gv, etc. Since the sampled data \
					  for computing Gauss-Newton matrix and etc. might not fit into memeory \
					  for one time, we will split the data into several segements and average\
					  over them.',
					  default=1024, type=int)
	parser.add_argument('--net', dest='net',
					  help='classifier type',
					  default='CNN_3layers', type=str)
	parser.add_argument('--train_set', dest='train_set',
					  help='provide the directory of .mat file for training',
					  default='data/cifar10.mat', type=str)
	parser.add_argument('--val_set', dest='val_set',
					  help='provide the directory of .mat file for validation',
					  default='data/cifar10.t.mat', type=str)
	parser.add_argument('--model', dest='log_name',
					  help='save log and model to directory model',
					  default='./log_and_model/logger.log', type=str)
	parser.add_argument('--print_log_only', dest='print_log_only',
					  help='screen printing running log instead of storing it',
					  action='store_true')
	parser.add_argument('--optim', '-optim', 
					  help='which optimizer to use: SGD, Adam or NewtonCG',
					  default='NewtonCG', type=str)
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

args = parse_args()

def init_model(param):
	init_ops = []
	for p in param:
		if 'kernel' in p.name:
			weight = np.random.standard_normal(p.shape)* np.sqrt(2.0 / ((np.prod(p.shape[:-1])).value))
			opt = tf.assign(p, weight)
		elif 'bias' in p.name:
			zeros = np.zeros(p.shape)
			opt = tf.assign(p, zeros)
		init_ops.append(opt)
	return tf.group(*init_ops)

def gradient_trainer(config, sess, network, full_batch, val_batch, test_network):
	x, y, loss, outputs,  = network
	
	global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
	learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

	# Probably not a good way to add regularization.
	# Just to confirm the implementation is the same as MATLAB.
	reg = 0.0
	param = tf.trainable_variables()
	for p in param:
		reg = reg + tf.reduce_sum(tf.pow(p,2))
	reg_const = 1/(2*config.C)
	loss_with_reg = reg_const*reg + loss/config.bsize

	if config.optim == 'SGD':
		optimizer = tf.train.MomentumOptimizer(
					learning_rate=learning_rate, 
					momentum=config.momentum).minimize(
					loss_with_reg, 
					global_step=global_step,
					colocate_gradients_with_ops=True)
	elif config.optim == 'Adam':
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
								beta1=0.9,
								beta2=0.999,
								epsilon=1e-08).minimize(
								loss_with_reg, 
								global_step=global_step,
								colocate_gradients_with_ops=True)

	train_inputs, train_labels = full_batch
	
	num_data = train_labels.shape[0]
	batch_size = config.bsize
	num_iters = math.ceil(num_data/batch_size)

	print(config.args)
	if not config.print_log_only:
		log_file = open(config.log_file, 'w')
		print(config.args, file=log_file)
	sess.run(tf.global_variables_initializer())
	

	print('-------------- initializing network by methods in He et al. (2015) --------------')
	param = tf.trainable_variables()
	sess.run(init_model(param))

	total_running_time = 0.0
	best_acc = 0.0

	saver = tf.train.Saver(var_list=param)

	for epoch in range(0, args.epoch):
		
		loss_avg = 0.0
		start = time.time()

		lr = config.lr * (0.1 ** (epoch // args.lr_decay))

		for i in range(num_iters):
			
			load_time = time.time()
			# shuffle training data
			idx = np.arange(0, num_data)
			np.random.shuffle(idx)
			idx = idx[:config.bsize]

			batch_input = train_inputs[idx]
			batch_labels = train_labels[idx]
			batch_input = np.ascontiguousarray(batch_input)
			batch_labels = np.ascontiguousarray(batch_labels)
			config.elapsed_time += time.time() - load_time

			_, _, batch_loss= sess.run(
				[global_step, optimizer, loss_with_reg],
				feed_dict = {x: batch_input, y: batch_labels, learning_rate: lr}
				)

			loss_avg = loss_avg + batch_loss
			# print log every 10% of the iterations
			if i % (num_iters//10) == 0:
				end = time.time()
				output_str = 'Epoch {}: {}/{} | loss {:.4f} | lr {:.6} | elapsed time {:.3f}'\
					.format(epoch, i, num_iters, batch_loss , lr, end-start)
				print(output_str)
				if not config.print_log_only:
					print(output_str, file=log_file)

		# exclude data loading time for fair comparison
		epoch_end = time.time() - config.elapsed_time
		total_running_time += epoch_end - start
		config.elapsed_time = 0.0
		
		if test_network == None:
			val_loss, val_acc = predict(
				sess, 
				network=(x, y, loss, outputs),
				test_batch=val_batch,
				bsize=config.bsize
				)
		else:
			val_loss, val_acc = predict(
				sess, 
				network=test_network,
				test_batch=val_batch,
				bsize=config.bsize
				)
		
		output_str = 'In epoch {} train loss: {:.3f} | val loss: {:.3f} | val accuracy: {:.3f}% | epoch time {:.3f}'\
			.format(epoch, loss_avg/(i+1), val_loss, val_acc*100, epoch_end-start)
		print(output_str)
		if not config.print_log_only:
			print(output_str, file=log_file)
		
		if val_acc > best_acc:
			best_acc = val_acc
			checkpoint_path = config.dir_name + '/best-model.ckpt' 
			save_path = saver.save(sess, checkpoint_path)
			print('Saved best model in {}'.format(save_path))

	output_str = 'Final acc: {:.3f}% | best acc {:.3f}% | total running time {:.3f}s'\
		.format(val_acc*100, best_acc*100, total_running_time)
	print(output_str)
	if not config.print_log_only:
		print(output_str, file=log_file)
		log_file.close()

def newton_trainer(config, sess, network, full_batch, val_batch, test_network):

	_, _, loss, outputs = network
	newton_solver = newton_cg(config, sess, outputs, loss)
	sess.run(tf.global_variables_initializer())

	print('-------------- initializing network by methods in He et al. (2015) --------------')
	param = tf.trainable_variables()
	sess.run(init_model(param))

	newton_solver.newton(full_batch, val_batch, network, test_network)


def main():

	full_batch = read_data(filename=args.train_set, num_cls=args.num_cls, dim=args.dim)
	val_batch = read_data(filename=args.val_set, num_cls=args.num_cls, dim=args.dim)

	num_data = full_batch[0].shape[0]

	config = Config(args, num_data)
	# tf.random.set_random_seed(0)
	# np.random.seed(0)
	
	if args.net in ('CNN_3layers', 'CNN_6layers'):
		x, y, outputs = CNN(config)
		test_network = None
	else:
		raise ValueError('Unrecognized training model')

	if config.loss == 'MSELoss':
		loss = tf.reduce_sum(tf.pow(outputs-y, 2))
	else:
		loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=y))
	
	network = (x, y, loss, outputs)

	sess_config = tf.ConfigProto()
	sess_config.gpu_options.allow_growth = False

	with tf.Session(config=sess_config) as sess:

		if config.optim in ('SGD', 'Adam'):
			gradient_trainer(
				config, sess, network, full_batch, val_batch, test_network)
		elif config.optim == 'NewtonCG':
			newton_trainer(
				config, sess, network, full_batch, val_batch, test_network=test_network)


if __name__ == '__main__':
	main()
