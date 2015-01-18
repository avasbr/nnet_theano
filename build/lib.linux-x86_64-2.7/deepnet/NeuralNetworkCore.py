import cPickle
import os
import sys
import time
import numpy as np
from deepnet.common import nnetutils as nu
from deepnet.common import nnetloss as nl
from deepnet.common import nnetact as na
from deepnet.common import nnetoptim as nopt
from deepnet.common import nneterror as ne
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class Network(object):
	''' Core neural network class that forms the basis for all further implementations (e.g. 
		MultilayerNet, Autoencoder, etc). Contains basic functions for propagating data forward
		and backwards through the network, as well as fitting the weights to data'''

	def __init__(self,d=None,k=None,num_hids=None,activs=None,loss_terms=[None],**loss_params):

		# Number of units in the output layer determined by k, so not explicitly specified in 
		# num_hids. still need to check that there's one less hidden unit sizes specified than
		# number of activation functions
		assert(len(num_hids)+1 == len(activs))

		# number of nodes
		self.num_nodes = [d]+num_hids+[k]

		# define activation functions		
		self.activs = [None]*len(activs)
		for idx,activ in enumerate(activs):
			if activ == 'sigmoid':
				self.activs[idx] = na.sigmoid
			elif activ == 'tanh':
				self.activs[idx] = na.tanh
			elif activ == 'reLU':
				self.activs[idx] = na.reLU
			elif activ == 'softmax':
				self.activs[idx] = na.softmax
			else:
				sys.exit(ne.activ_err())

		# loss function and parameters
		self.loss_terms = loss_terms
		self.loss_params = loss_params
		
		# initialize the random number stream
		self.srng = RandomStreams() 

	def set_weights(self,wts=None,bs=None,init_method='gauss',scale_factor=0.001):
		''' Initializes the weights and biases of the neural network 
		
		Parameters:
		-----------
		param: wts - weights
		type: np.ndarray, optional

		param: bs - biases
		type: np.ndarray, optional

		param: method - calls some pre-specified weight initialization routines
		type: string, optional
		'''
		# weights and biases
		if wts is None and bs is None:
			wts = (len(self.num_nodes)-1)*[None]
			bs = (len(self.num_nodes)-1)*[None]
			
			if init_method == 'gauss':
				for i,(n1,n2) in enumerate(zip(self.num_nodes[:-1],self.num_nodes[1:])):
					wts[i] = scale_factor*np.random.randn(n1,n2)
					bs[i] = np.zeros(n2)

			if init_method == 'fan-io':
				for i,(n1,n2) in enumerate(zip(self.num_nodes[:-1],self.num_nodes[1:])):
					v = np.sqrt(scale_factor*1./(n1+n2+1))
					wts[i] = 2.0*v*np.random.rand(n1,n2)-v 
					bs[i] = np.zeros(n2)
		else:
			assert isinstance(wts,list)
			assert isinstance(bs,list)
			
		self.wts_ = [theano.shared(nu.floatX(w),borrow=True) for w in wts]
		self.bs_ = [theano.shared(nu.floatX(b),borrow=True) for b in bs]

	def fit(self,X_tr,y_tr,X_val=None,y_val=None,**optim_params):
		''' The primary function which ingests data and fits to the neural network. Currently
		only supports mini-batch training.

		Parameters:
		-----------
		param: X_tr - training data
		type: theano matrix

		param: y_tr - training labels
		type: theano matrix

		param: X_val - validation data
		type: theano matrix

		param: y_val - validation labels
		type: theano matrix

		param: **optim_params
		type: dictionary of optimization parameters 

		'''
		# initialize all the weights
		if all(node for node in self.num_nodes):
			init_method = optim_params.pop('init_method')
			scale_factor = optim_params.pop('scale_factor')
			self.set_weights(init_method=init_method,scale_factor=scale_factor)

		# get the method and learning type
		try:
			optim_method = optim_params.pop('optim_method')
		except KeyError:
			sys.exit(ne.method_err())

		try:
			optim_type = optim_params.pop('optim_type')
		except KeyError:
			sys.exit(ne.type_err())

		# perform minibatch or full-batch optimization
		num_epochs = optim_params.pop('num_epochs',None)
		batch_size = optim_params.pop('batch_size',None)

		# get the expressions for computing the training and validation losses, 
		# as well as the gradients

		X = T.matrix('X') # input variable
		y = T.matrix('y') # output variable
		
		optim_loss, eval_loss = self.compute_loss(X,y) # loss functions
		params = [p for param in [self.wts_,self.bs_] for p in param] # all model parameters in a list
		grad_params = [T.grad(optim_loss,param) for param in params] # gradient of each model param w.r.t training loss
		
		# define the update rule 
		updates = []
		if optim_method == 'SGD':
			updates = nopt.sgd(params,grad_params,**optim_params) # update rule
		
		elif optim_method == 'ADAGRAD':
			updates = nopt.adagrad(params,grad_params,**optim_params) # update rule
		
		elif optim_method == 'RMSPROP':
			updates = nopt.rmsprop(params,grad_params,**optim_params)
		
		else:
			print method_err()

		# compile the training and loss computation functions
		self.train = theano.function(inputs=[X,y],updates=updates,allow_input_downcast=True,
			mode='FAST_RUN')

		self.compute_optim_loss = theano.function(inputs=[X,y],outputs=optim_loss,allow_input_downcast=True,
			mode='FAST_RUN')

		self.compute_eval_loss = theano.function(inputs=[X,y],outputs=eval_loss,allow_input_downcast=True,
			mode='FAST_RUN')

		if optim_type == 'minibatch':	
			# mini-batch optimization
			self.minibatch_optimize(X_tr,y_tr,X_val=X_val,y_val=y_val,batch_size=batch_size,num_epochs=num_epochs)
		elif optim_type == 'fullbatch':
			# full-batch optimization
			self.fullbatch_optimize(X_tr,y_tr,X_val=X_val,y_val=y_val,num_epochs=num_epochs)
		else:
			# error
			sys.exit(type_err())
		
		return self

	def shared_dataset(self,X,y):
		''' As per the deep learning tutorial in theano, loading the data all at once (if possible)
		into the GPU will significantly speed things up '''

		return theano.shared(nu.floatX(X)),theano.shared(nu.floatX(y))

	def fullbatch_optimize(self,X_tr,y_tr,X_val=None,y_val=None,num_epochs=500):
		''' Full-batch optimization using update functions 

		Parameters:
		-----------
		param: X_tr - training data
		type: theano matrix

		param: y_tr - training labels
		type: theano matrix

		param: num_epochs - the number of full runs through the dataset
		type: int
		'''
		pass

	def minibatch_optimize(self,X_tr,y_tr,X_val=None,y_val=None,batch_size=100,num_epochs=500):
		''' Mini-batch optimization using update functions 

		Parameters:
		-----------
		param: X_tr - training data
		type: theano matrix

		param: y_tr - training labels
		type: theano matrix

		param: batch_size - number of examples per mini-batch
		type: int

		param: num_epochs - the number of full runs through the dataset
		type: int
		'''

		# define the mini-batches
		m = X_tr.shape[0] # total number of training instances
		n_batches = int(m/batch_size) # number of batches, based on batch size
		leftover = m-n_batches*batch_size # batch_size won't divide the data evenly, so get leftover
		epoch = 0

		X_tr,y_tr = self.shared_dataset(X_tr,y_tr)

		# iterate through the training examples
		while epoch < num_epochs:
			epoch += 1
			tr_idx = np.random.permutation(m) # randomly shuffle the data indices
			ss_idx = range(0,m,batch_size)
			ss_idx[-1] += leftover # add the leftovers to the last batch
			
			# run through a full epoch
			for idx,(start_idx,stop_idx) in enumerate(zip(ss_idx[:-1],ss_idx[1:])):			
				
				n_batch_iter = (epoch-1)*n_batches + idx # total number of batches processed up until now
				batch_idx = tr_idx[start_idx:stop_idx] # get the next batch
				
				self.train(X_tr[batch_idx,:],y_tr[batch_idx,:]) # update the model
				
			if epoch%10 == 0:
				tr_loss = self.compute_eval_loss(X_tr,y_tr)
				print 'Epoch: %s, Training error: %.3f'%(epoch,tr_loss)

	def dropout(self,act,p=0.5):
		''' Randomly drops an activation with probability p 
		
		Parameters
		----------
		param: act - activation values, in a matrix
		type: theano matrix

		param: p - probability of dropping out a node
		type: float, optional

		Returns:
		--------
		param: [expr] - activation values randomly zeroed out
		type: theano matrix

		'''
		if p > 0:
			# randomly dropout p activations 
			retain_prob = 1.-p
			return (1./retain_prob)*act*self.srng.binomial(act.shape,p=retain_prob,dtype=theano.config.floatX)

	def dropout_fprop(self,X,wts=None,bs=None):
		''' Performs forward propagation with dropout
		
		Parameters:
		-----------
		param: X - input data
		type: theano matrix

		param: wts - weights
		type: numpy ndarray, optional

		param: bs - bias
		type: numpy ndarray, optional

		Returns:
		--------
		param: final activation values
		type: theano matrix
	
		'''
		
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		# get the input and hidden layer dropout probabilities
		input_p = self.loss_params['input_p']
		hidden_p = self.loss_params['hidden_p']
		
		act = self.activs[0](T.dot(self.dropout(X,input_p),wts[0]) + bs[0]) # compute the first activation
		if len(wts) > 1: # len(wts) = 1 corresponds to softmax regression
			for i,(w,b,activ) in enumerate(zip(wts[1:],bs[1:],self.activs[1:])):
				act = activ(T.dot(self.dropout(act,hidden_p),w) + b)

		return act

	def fprop(self,X,wts=None,bs=None):
		''' Performs forward propagation through the network

		Parameters
		----------
		param: X - training data
		type: theano matrix

		param: wts - weights
		type: theano matrix

		param: bs - biases
		type: theano matrix

		Returns:
		--------
		param: act - final activation values
		type: theano matrix
		'''
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		act = self.activs[0](T.dot(X,wts[0]) + bs[0]) # use the first data matrix to compute the first activation
		
		if len(wts) > 1:
			for i,(w,b,activ) in enumerate(zip(wts[1:],bs[1:],self.activs[1:])):
				act = activ(T.dot(act,w) + b)

		return act

	def compute_loss(self,X,y,wts=None,bs=None):
		''' Given inputs, returns the loss at the current state of the model

		Parameters:
		-----------
		param: X - training data
		type: theano matrix

		param: y - training labels
		type: theano matrix

		param: wts - weights
		type: theano matrix, optional

		param: bs - biases
		type: theano matrix, optional

		Returns:
		--------
		param: optim_loss - the optimization loss which must be optimized over
		type: theano scalar

		param: eval_loss - evaluation loss, which doesn't include regularization
		type: theano scalar

		'''
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_
		
		if 'dropout' in self.loss_terms:
			y_optim = self.dropout_fprop(X,wts,bs) # based on the output from applying dropout
		else:
			y_optim = self.fprop(X,wts,bs)

		y_pred = self.fprop(X,wts,bs)

		optim_loss = None # the loss function which will specifically be optimized over
		eval_loss = None # the loss function we can evaluate during validation

		if 'cross_entropy' in self.loss_terms:
			optim_loss = nl.cross_entropy(y,y_optim)
			eval_loss = nl.cross_entropy(y,y_pred)
		
		elif 'squared_loss' in self.loss_terms:
			optim_loss = nl.squared_loss(y,y_pred)
			eval_loss = nl.cross_entropy(y,y_pred)
		
		else:
			sys.exit('Must be either cross_entropy or squared_loss')

		if 'regularization' in self.loss_terms:
			
			L1_decay = self.loss_params.get('L1_decay')
			L2_decay = self.loss_params.get('L2_decay')

			optim_loss += nl.regularization(wts,L1_decay=L1_decay,L2_decay=L2_decay)
			
		return optim_loss,eval_loss