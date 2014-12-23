import cPickle
import os
import sys
import time
import numpy as np
import nnetutils as nu
import nnetoptim as nopt
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class Network(object):

	def __init__(self,d=None,k=None,num_hid=None,activ=None,dropout_flag=False,input_p=None,
		hidden_p=None,loss_func=None,**loss_params):

		# network parameters
		self.num_nodes = [d]+num_hid+[k] # number of nodes
		self.activ = activ
				
		if all(node for node in self.num_nodes):
			self.set_weights(method='gauss')
		
		# loss function and parameters
		self.loss_func = loss_func
		self.loss_params = loss_params
		
		# dropout
		self.dropout_flag = dropout_flag 
		
		# add defensive coding here
		self.input_p = input_p
		self.hidden_p = hidden_p
		self.srng = RandomStreams() # initialize the random number stream

	def set_weights(self,wts=None,bs=None,method='gauss'):
		''' Initializes the weights and biases of the neural network '''

		# weights and biases
		if wts is None and bs is None:
			self.wts_ = (len(self.num_nodes)-1)*[None]
			self.bs_ = (len(self.num_nodes)-1)*[None]
			
			if method == 'gauss':
				for i,(n1,n2) in enumerate(zip(self.num_nodes[:-1],self.num_nodes[1:])):
					self.wts_[i] = theano.shared(nu.floatX(0.01*np.random.randn(n1,n2)))
					self.bs_[i] = theano.shared(nu.floatX(np.zeros(n2)))

			if method == 'random':
				for i,(n1,n2) in enumerate(zip(self.num_nodes[:-1],self.num_nodes[1:])):
					v = 1.*np.sqrt(6./(n1+n2+1))
					self.wts_[i] = theano.shared(nu.floatX(2.0*v*np.random.rand(n1,n2)-v)) 
					self.bs_[i] = theano.shared(nu.floatX(np.zeros(n2)))
		else:
			assert isinstance(wts,list)
			assert isinstance(bs,list)
			
			self.wts_ = [theano.shared(nu.floatX(w)) for w in wts]
			self.bs_ = [theano.shared(nu.floatX(b)) for b in bs]

	def fit(self,X_tr,y_tr,X_val=None,y_val=None,**optim_params):
		''' The main function which pulls everything together to train the neural network

		param: X_tr - training data
		type: theano matrix

		param: y_tr - training labels
		type: theano matrix

		param: X_val - validation data
		type: theano matrix

		param: y_val - validation labels
		type: theano matrix

		param: **optim_params
		type: dictionary of optimization 

		'''
		# get the method and relevant 
		def method_err():
			err_msg = ('No method provided to fit! Your choices are:'
						'\n(1) SGD: stochastic gradient descent with optional (improved/nesterov) momentum'+
						'\n(2) ADAGRAD: ADAptive GRADient learning'+
						'\n(3) RMSPROP: Hintons mini-batch version of RPROP [NOT IMPLEMENTED]')

			return err_msg

		# get the method, batch size, and number of epochs
		try:
			method = optim_params.pop('method')
		except KeyError:
			sys.exit(method_err())

		batch_size = optim_params.pop('batch_size',None)
		num_epochs = optim_params.pop('num_epochs',None)

		# get the expressions for computing the training and validation losses, as well as the gradients

		X = T.matrix('X') # input variable
		y = T.matrix('y') # output variable
		
		optim_loss, eval_loss = self.compute_loss(X,y,self.wts_,self.bs_) # loss functions
		params = [p for param in [self.wts_,self.bs_] for p in param] # all model parameters
		grad_params = [T.grad(optim_loss,param) for param in params] # gradient of each model param w.r.t training loss
		
		# define the update rule 
		updates = []
		if method == 'SGD':
			updates = nopt.stochastic_gradient_descent(params,grad_params,**optim_params) # update rule
		
		elif method == 'ADAGRAD':
			updates = nopt.adagrad(params,grad_params,**optim_params) # update rule
		
		elif method == 'RMSPROP':
			updates = nopt.rmsprop(params,grad_params,**optim_params)
		
		else:
			print method_err()

		self.train = theano.function(inputs=[X,y],updates=updates,allow_input_downcast=True,
			mode='FAST_RUN')

		self.compute_optim_loss = theano.function(inputs=[X,y],outputs=optim_loss,allow_input_downcast=True,
			mode='FAST_RUN')

		self.compute_eval_loss = theano.function(inputs=[X,y],outputs=eval_loss,allow_input_downcast=True,
			mode='FAST_RUN')

		# minibatch optimization 
		self.minibatch_optimize(X_tr,y_tr,batch_size=batch_size,num_epochs=num_epochs)

		return self

	def minibatch_optimize(self,X_tr,y_tr,batch_size=100,num_epochs=500):
		''' Mini-batch optimization using update functions 

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

	def dropout(self,act,p=0):
		''' Randomly drops an activation with probability p '''
		if p > 0:
			# randomly dropout p activations 
			retain_prob = 1.-p
			return (1./retain_prob)*act*self.srng.binomial(act.shape,p=retain_prob,dtype=theano.config.floatX)

	def dropout_fprop(self,X,wts=None,bs=None):
		''' forward propagation with dropout, for training '''
		
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		act = self.activ[0](T.dot(self.dropout(X,self.input_p),wts[0]) + bs[0]) # compute the first activation
		if len(wts) > 1: # len(wts) = 1 corresponds to softmax regression
			for i,(w,b,activ) in enumerate(zip(wts[1:],bs[1:],self.activ[1:])):
				act = activ(T.dot(self.dropout(act,self.hidden_p),w) + b)

		return act

	def fprop(self,X,wts=None,bs=None):
		''' Performs forward propagation through the network, and updates all intermediate values. if
		input_dropout and hidden_dropout are specified, uses those to randomly omit hidden units - very 
		powerful technique'''

		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_
		
		act = self.activ[0](T.dot(X,wts[0]) + bs[0]) # use the first data matrix to compute the first activation
		
		if len(wts) > 1:
			for i,(w,b,activ) in enumerate(zip(wts[1:],bs[1:],self.activ[1:])):
				act = activ(T.dot(act,w) + b)

		return act

	def compute_loss(self,X,y,wts=None,bs=None):
		''' Given inputs (X,y), returns the loss at the current state of the model (wts,bs) '''
		
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_
		
		if self.dropout_flag:
			# the optimization loss is based on the output from applying dropout
			y_prob = self.dropout_fprop(X,wts,bs)
			optim_loss = self.loss_func(y,y_prob) + nu.regularization(wts)

			# the evaluation loss is based on the expected value of the weights 
			# y_prob = self.dropout_eval_fprop(X,wts,bs)
			y_prob = self.fprop(X,wts,bs)
			eval_loss = self.loss_func(y,y_prob)

		else:
			y_prob = self.fprop(X,wts,bs)
			eval_loss = self.loss_func(y,y_prob) # this is the loss that will be used to evaluate any set
			optim_loss = self.loss_func(y,y_prob) + nu.regularization(wts) # this is the loss which will be specifically optimized over

		return optim_loss,eval_loss