import pdb
import tensorflow as tf
import time
import numpy as np
import os
import math
from utilities import read_data, predict

class Config(object):
	def __init__(self, args, num_data):
		super(Config, self).__init__()
		self.args = args
		# self.sample = args.sample
		self.iter_max = args.iter_max
		
		# Different notations of regularization term:
		# In SGD, weight decay:
		# 	weight_decay <- lr/(C*num_of_training_samples)
		# In Newton method:
		# 	C <- C * num_of_training_samples

		self.train_set = args.train_set
		self.val_set = args.val_set
		self.num_cls = args.num_cls
		self.dim = args.dim

		self.num_data = num_data
		self.sample = min(args.sample, self.num_data)
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
		
		self.log_file = args.log_name
		self.model_name = args.model_name
		self.screen_log_only = args.screen_log_only
		dir_name, _ = os.path.split(self.log_file)
		self.dir_name = dir_name

		if self.screen_log_only:
			print('You choose not to store running log. Only store model to {}'.format(self.log_file))
		else:
			print('Saving log and model to: {}'.format(self.model_name))
		if not os.path.isdir(dir_name):
			os.makedirs(dir_name, exist_ok=True)
		
		self.elapsed_time = 0.0


def Rop(f, weights, v):
	"""Implementation of R operator
	Args:
		f: any function of param
		weights: Weights, list of tensors.
		v: vector for right multiplication
	Returns:
		Jv: Jaccobian vector product, same size, same shape as vector v.
	"""
	if type(f) == list:
		u = [tf.ones_like(ff) for ff in f]
	else:
		u = tf.ones_like(f)  # dummy variable
	g = tf.gradients(f, weights, grad_ys=u)
	return tf.gradients(g, u, grad_ys=v)

def Gauss_Newton_vec(outputs, loss, weights, v):
	"""Implements Gauss-Newton vector product.
	Args:
		loss: Loss function.
		outputs: outputs of the last layer (pre-softmax).
		weights: Weights, list of tensors.
		v: vector to be multiplied with Gauss Newton matrix
	Returns:
		J'BJv: Guass-Newton vector product.
	"""
	# Validate the input
	if type(weights) == list:
		if len(v) != len(weights):
			raise ValueError("weights and v must have the same length.")

	grads_outputs = tf.gradients(loss, outputs)
	BJv = Rop(grads_outputs, weights, v)
	JBJv = tf.gradients(outputs, weights, BJv)
	return JBJv
	

class newton_cg(object):
	def __init__(self, config, sess, outputs, loss):
		"""
		initialize operations and vairables that will be used in newton
		args:
			sess: tensorflow session
			outputs: output of the neural network (pre-softmax layer)
			loss: function to calculate loss
		"""
		super(newton_cg, self).__init__()
		self.sess = sess
		self.config = config
		self.outputs = outputs
		self.loss = loss
		self.param = tf.trainable_variables()

		self.CGiter = 0
		FLOAT = tf.float32
		model_weight = self.vectorize(self.param)
		
		# initial variable used in CG
		zeros = tf.zeros(model_weight.get_shape(), dtype=FLOAT)
		self.r = tf.Variable(zeros, dtype=FLOAT, trainable=False)
		self.v = tf.Variable(zeros, dtype=FLOAT, trainable=False)
		self.s = tf.Variable(zeros, dtype=FLOAT, trainable=False)
		self.g = tf.Variable(zeros, dtype=FLOAT, trainable=False)
		# initial Gv, f for method minibatch
		self.Gv = tf.Variable(zeros, dtype=FLOAT, trainable=False)
		self.f = tf.Variable(0., dtype=FLOAT, trainable=False)

		# rTr, cgtol and beta to be used in CG
		self.rTr = tf.Variable(0., dtype=FLOAT, trainable=False)
		self.cgtol = tf.Variable(0., dtype=FLOAT, trainable=False)
		self.beta = tf.Variable(0., dtype=FLOAT, trainable=False)

		# placeholder alpha, old_alpha and lambda
		self.alpha = tf.placeholder(FLOAT, shape=[])
		self.old_alpha = tf.placeholder(FLOAT, shape=[])
		self._lambda = tf.placeholder(FLOAT, shape=[])

		self.num_grad_segment = math.ceil(self.config.num_data/self.config.bsize)
		self.num_Gv_segment = math.ceil(self.config.sample/self.config.bsize)

		cal_loss, cal_lossgrad, cal_lossGv, \
		add_reg_avg_loss, add_reg_avg_grad, add_reg_avg_Gv, \
		zero_loss, zero_grad, zero_Gv = self._ops_in_minibatch()

		# initial operations that will be used in minibatch and newton
		self.cal_loss = cal_loss
		self.cal_lossgrad = cal_lossgrad
		self.cal_lossGv = cal_lossGv
		self.add_reg_avg_loss = add_reg_avg_loss
		self.add_reg_avg_grad = add_reg_avg_grad
		self.add_reg_avg_Gv = add_reg_avg_Gv
		self.zero_loss = zero_loss
		self.zero_grad = zero_grad
		self.zero_Gv = zero_Gv

		self.CG, self.update_v = self._CG()
		self.init_cg_vars = self._init_cg_vars()
		self.update_gs = tf.tensordot(self.s, self.g, axes=1)
		self.update_sGs = 0.5*tf.tensordot(self.s, -self.g-self.r-self._lambda*self.s, axes=1)
		self.update_model = self._update_model()
		self.gnorm = self.calc_norm(self.g)


	def vectorize(self, tensors):
		if isinstance(tensors, list) or isinstance(tensors, tuple):
			vector = [tf.reshape(tensor, [-1]) for tensor in tensors]
			return tf.concat(vector, 0) 
		else:
			return tensors 
	
	def inverse_vectorize(self, vector, param):
		if isinstance(vector, list):
			return vector
		else:
			tensors = []
			offset = 0
			num_total_param = np.sum([np.prod(p.shape.as_list()) for p in param])
			for p in param:
				numel = np.prod(p.shape.as_list())
				tensors.append(tf.reshape(vector[offset: offset+numel], p.shape))
				offset += numel

			assert offset == num_total_param
			return tensors

	def calc_norm(self, v):
		# default: frobenius norm
		if isinstance(v, list):
			norm = 0.
			for p in v:
				norm = norm + tf.norm(p)**2
			return norm**0.5
		else:
			return tf.norm(v)

	def _ops_in_minibatch(self):
		"""
		Define operations that will be used in method minibatch

		Vectorization is already a deep copy operation.
		Before using newton method, loss needs to be summed over training samples
		to make results consistent.
		"""

		def cal_loss():
			return tf.assign(self.f, self.f + self.loss)

		def cal_lossgrad():
			update_f = tf.assign(self.f, self.f + self.loss)

			grad = tf.gradients(self.loss, self.param)
			grad = self.vectorize(grad)
			update_grad = tf.assign(self.g, self.g + grad)

			return tf.group(*[update_f, update_grad])

		def cal_lossGv():
			v = self.inverse_vectorize(self.v, self.param)
			Gv = Gauss_Newton_vec(self.outputs, self.loss, self.param, v)
			Gv = self.vectorize(Gv)
			return tf.assign(self.Gv, self.Gv + Gv) 

		# add regularization term to loss, gradient and Gv and further average over batches 
		def add_reg_avg_loss():
			model_weight = self.vectorize(self.param)
			reg = (self.calc_norm(model_weight))**2
			reg = 1.0/(2*self.config.C) * reg
			return tf.assign(self.f, reg + self.f/self.config.num_data)

		def add_reg_avg_lossgrad():
			model_weight = self.vectorize(self.param)
			reg_grad = model_weight/self.config.C
			return tf.assign(self.g, reg_grad + self.g/self.config.num_data)

		def add_reg_avg_lossGv():
			return tf.assign(self.Gv, (self._lambda + 1/self.config.C)*self.v
			 + self.Gv/self.config.sample) 

		# zero out loss, grad and Gv 
		def zero_loss():
			return tf.assign(self.f, tf.zeros_like(self.f))
		def zero_grad():
			return tf.assign(self.g, tf.zeros_like(self.g))
		def zero_Gv():
			return tf.assign(self.Gv, tf.zeros_like(self.Gv))

		return (cal_loss(), cal_lossgrad(), cal_lossGv(),
				add_reg_avg_loss(), add_reg_avg_lossgrad(), add_reg_avg_lossGv(),
				zero_loss(), zero_grad(), zero_Gv())

	def minibatch(self, data_batch, place_holder_x, place_holder_y, mode):
		"""
		A function to evaluate either function value, global gradient or sub-sampled Gv
		"""
		if mode not in ('funonly', 'fungrad', 'Gv'):
			raise ValueError('Unknown mode other than funonly & fungrad & Gv!')

		inputs, labels = data_batch
		num_data = labels.shape[0]
		num_segment = math.ceil(num_data/self.config.bsize)
		x, y = place_holder_x, place_holder_y

		# before estimation starts, need to zero out f, grad and Gv according to the mode

		if mode == 'funonly':
			assert num_data == self.config.num_data
			assert num_segment == self.num_grad_segment
			self.sess.run(self.zero_loss)
		elif mode == 'fungrad':
			assert num_data == self.config.num_data
			assert num_segment == self.num_grad_segment
			self.sess.run([self.zero_loss, self.zero_grad])
		else:
			assert num_data == self.config.sample
			assert num_segment == self.num_Gv_segment
			self.sess.run(self.zero_Gv)

		for i in range(num_segment):
			
			load_time = time.time()
			idx = np.arange(i * self.config.bsize, min((i+1) * self.config.bsize, num_data))
			batch_input = inputs[idx]
			batch_labels = labels[idx]
			batch_input = np.ascontiguousarray(batch_input)
			batch_labels = np.ascontiguousarray(batch_labels)
			self.config.elapsed_time += time.time() - load_time

			if mode == 'funonly':

				self.sess.run(self.cal_loss, feed_dict={
							x: batch_input, 
							y: batch_labels,})

			elif mode == 'fungrad':
				
				self.sess.run(self.cal_lossgrad, feed_dict={
							x: batch_input, 
							y: batch_labels,})
				
			else:
				
				self.sess.run(self.cal_lossGv, feed_dict={
							x: batch_input, 
							y: batch_labels})

		# average over batches
		if mode == 'funonly':
			self.sess.run(self.add_reg_avg_loss)
		elif mode == 'fungrad':
			self.sess.run([self.add_reg_avg_loss, self.add_reg_avg_grad])
		else:
			self.sess.run(self.add_reg_avg_Gv, 
				feed_dict={self._lambda: self.config._lambda})


	def _update_model(self):
		update_model_ops = []
		x = self.inverse_vectorize(self.s, self.param)
		for i, p in enumerate(self.param):
			op = tf.assign(p, p + (self.alpha-self.old_alpha) * x[i])
			update_model_ops.append(op)
		return tf.group(*update_model_ops)

	def _init_cg_vars(self):
		init_ops = []

		init_r = tf.assign(self.r, -self.g)
		init_v = tf.assign(self.v, -self.g)
		init_s = tf.assign(self.s, tf.zeros_like(self.g))
		gnorm = self.calc_norm(self.g)
		init_rTr = tf.assign(self.rTr, gnorm**2)
		init_cgtol = tf.assign(self.cgtol, self.config.xi*gnorm)

		init_ops = [init_r, init_v, init_s, init_rTr, init_cgtol]

		return tf.group(*init_ops)

	def _CG(self):
		"""
		CG:
			define operations that will be used in method newton

		Same as the previous loss calculation,
		Gv has been summed over batches when samples were fed into Neural Network.
		"""

		def CG_ops():
			
			vGv = tf.tensordot(self.v, self.Gv, axes=1)

			alpha = self.rTr / vGv
			with tf.control_dependencies([alpha]):
				update_s = tf.assign(self.s, self.s + alpha * self.v, name='update_s_ops')
				update_r = tf.assign(self.r, self.r - alpha * self.Gv, name='update_r_ops')

				with tf.control_dependencies([update_s, update_r]):
					rnewTrnew = self.calc_norm(update_r)**2
					update_beta = tf.assign(self.beta, rnewTrnew / self.rTr)
					with tf.control_dependencies([update_beta]):
						update_rTr = tf.assign(self.rTr, rnewTrnew, name='update_rTr_ops')

			return tf.group(*[update_s, update_beta, update_rTr])

		def update_v():
			return tf.assign(self.v, self.r + self.beta*self.v, name='update_v')

		return (CG_ops(), update_v())


	def newton(self, full_batch, val_batch, network, test_network=None):
		"""
		Conduct newton steps for training
		args:
			full_batch & val_batch: provide training set and validation set. The function will
				save the best model evaluted on validation set for future prediction.
			network: a tuple contains (x, y, loss, outputs).
			test_network: a tuple similar to argument network. If you use layers which behave differently
				in test phase such as batchnorm, a separate test_network is needed.
		return:
			None
		"""
		# check whether data is valid
		full_inputs, full_labels = full_batch
		assert full_inputs.shape[0] == full_labels.shape[0]

		if full_inputs.shape[0] != self.config.num_data:
			raise ValueError('The number of full batch inputs does not agree with the config argument.\
							This is important because global loss is averaged over those inputs')

		x, y, _, outputs = network

		tf.summary.scalar('loss', self.f)
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter('./summary/train', self.sess.graph)

		print(self.config.args)
		if not self.config.screen_log_only:
			log_file = open(self.config.log_file, 'w')
			print(self.config.args, file=log_file)
		
		self.minibatch(full_batch, x, y, mode='fungrad')
		f = self.sess.run(self.f)

		best_acc = 0.0

		total_running_time = 0.0

		saver = tf.train.Saver(var_list=self.param)

		for k in range(self.config.iter_max):

			idx = np.arange(0, full_labels.shape[0])
			np.random.shuffle(idx)
			idx = idx[:self.config.sample]
			mini_inputs = full_inputs[idx]
			mini_labels = full_labels[idx]

			start = time.time()

			self.sess.run(self.init_cg_vars)
			cgtol = self.sess.run(self.cgtol)

			for CGiter in range(1, self.config.CGmax+1):

				self.minibatch((mini_inputs, mini_labels), x, y, mode='Gv')
				
				self.sess.run(self.CG)

				rnewTrnew = self.sess.run(self.rTr)
				
				if rnewTrnew**0.5 <= cgtol or CGiter == self.config.CGmax:
					break

				self.sess.run(self.update_v)

			gs, sGs = self.sess.run([self.update_gs, self.update_sGs], feed_dict={
					self._lambda: self.config._lambda
				})
			
			# line_search
			f_old = f
			alpha = 1
			while True:

				old_alpha = 0 if alpha == 1 else alpha/0.5
				
				self.sess.run(self.update_model, feed_dict={
					self.alpha:alpha, self.old_alpha:old_alpha
					})

				prered = alpha*gs + (alpha**2)*sGs

				self.minibatch(full_batch, x, y, mode='funonly')
				f = self.sess.run(self.f)

				actred = f - f_old

				if actred <= self.config.eta*alpha*gs:
					break

				alpha *= 0.5

			# update lambda
			ratio = actred / prered
			if ratio < 0.25:
				self.config._lambda *= self.config.boost
			elif ratio >= 0.75:
				self.config._lambda *= self.config.drop

			self.minibatch(full_batch, x, y, mode='fungrad')
			f = self.sess.run(self.f)

			gnorm = self.sess.run(self.gnorm)

			summary = self.sess.run(merged)
			train_writer.add_summary(summary, k)

			# exclude data loading time for fair comparison
			end = time.time() - self.config.elapsed_time
			total_running_time += end-start

			self.config.elapsed_time = 0.0
			
			output_str = '{}-iter f: {:.3f} |g|: {:.5f} alpha: {:.3e} ratio: {:.3f} lambda: {:.5f} #CG: {} actred: {:.5f} prered: {:.5f} time: {:.3f}'.\
							format(k, f, gnorm, alpha, actred/prered, self.config._lambda, CGiter, actred, prered, end-start)
			print(output_str)
			if not self.config.screen_log_only:
				print(output_str, file=log_file)

			if val_batch is not None:
				# Evaluate the performance after every Newton Step
				if test_network == None:
					val_loss, val_acc = predict(
						self.sess, 
						network=(x, y, self.loss, outputs),
						test_batch=val_batch,
						bsize=self.config.bsize,
						)
				else:
					val_loss, val_acc = predict(
						self.sess, 
						network=test_network,
						test_batch=val_batch,
						bsize=self.config.bsize
						)

				output_str = '\r\n {}-iter val_acc: {:.3f}% val_loss {:.3f}\r\n'.\
					format(k, val_acc*100, val_loss)
				print(output_str)
				if not self.config.screen_log_only:
					print(output_str, file=log_file)

				if val_acc > best_acc:
					best_acc = val_acc
					checkpoint_path = self.config.model_name
					save_path = saver.save(self.sess, checkpoint_path)
					print('Best model saved in {}\r\n'.format(save_path))

		if val_batch is None:
			checkpoint_path = self.config.model_name
			save_path = saver.save(self.sess, checkpoint_path)
			print('Model at the last iteration saved in {}\r\n'.format(save_path))
			output_str = 'total running time {:.3f}s'.format(total_running_time)
		else:
			output_str = 'Final acc: {:.3f}% | best acc {:.3f}% | total running time {:.3f}s'.\
				format(val_acc*100, best_acc*100, total_running_time)
		print(output_str)
		if not self.config.screen_log_only:
			print(output_str, file=log_file)
			log_file.close()

