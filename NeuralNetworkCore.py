import cPickle
import os
import sys
import time
import numpy as np
import nnetutils as nu
import nnetoptim as nopt
import theano
import theano.tensor as T

class Network(object):

	def __init__(self,d=None,k=None,n_hid=None,activ=None,loss_func=None,**loss_params):

		# network parameters
		self.n_nodes = [d]+n_hid+[k] # number of nodes
		self.act = (len(self.n_nodes)-1)*[None]
		self.activ = activ
				
		if all(node for node in self.n_nodes):
			self.set_weights(method='random')
		
		self.loss_func = loss_func
		self.loss_params = loss_params

	def set_weights(self,wts=None,bs=None,method='random'):
		''' Initializes the weights and biases of the neural network '''

		# weights and biases
		if wts is None and bs is None:
			self.wts_ = (len(self.n_nodes)-1)*[None]
			self.bs_ = (len(self.n_nodes)-1)*[None]
			
			if method == 'random':
				for i,(n1,n2) in enumerate(zip(self.n_nodes[:-1],self.n_nodes[1:])):
					v = 4.*np.sqrt(6./(n1+n2+1))
					self.wts_[i] = theano.shared(nu.floatX(2.0*v*np.random.rand(n1,n2)-v)) 
					self.bs_[i] = theano.shared(nu.floatX(np.zeros(n2)))
			
			# fixed weights, mainly for debugging purposes
			else:
				for i,(n1,n2) in enumerate(zip(self.n_nodes[:-1],self.n_nodes[1:])):
					self.wts_[i] = theano.shared(nu.floatX(np.reshape(0.1*np.range(n1*n2),n1,n2)))
					self.bs_[i] = theano.shared(nu.floatX(np.zeros(n2)))
		else:
			assert isinstance(wts,list)
			assert isinstance(bs,list)
			
			self.wts_ = [theano.shared(nu.floatX(w)) for w in wts]
			self.bs_ = [theano.shared(nu.floatX(b)) for b in bs]

	def fit(self,X,y,X_val=None,y_val=None,**optim_params):
		''' '''

		def method_err():
			err_msg = ('No method provided to fit! Your choices are:'
						'\n(1) SGD: stochastic gradient descent'+
						'\n(2) ADAGRAD: ADAptive GRADient learning'+
						'\n(3) RMSPROP: Hintons mini-batch version of RPROP [NOT IMPLEMENTED]')
			return err_msg

		if 'method' not in optim_params or optim_params['method'] is None:
			sys.exit(method_err())
		
		method = optim_params['method']
		del optim_params['method']

		if method == 'SGD':
			nopt.minibatch_gradient_descent(X,y,self.wts_,self.bs_,self.compute_loss,
				self.compute_grad,**optim_params)

		elif method == 'ADAGRAD':
			nopt.adagrad(X,y,self.n_nodes,self.wts_,self.bs_,
				self.compute_loss,self.compute_grad,**optim_params)
		
		elif method == 'RMSPROP':
			pass
		
		else:
			print method_err()

		return self

	def fprop(self,X,wts=None,bs=None):
		''' Performs forward propagation through the network, and updates all intermediate values. if
		input_dropout and hidden_dropout are specified, uses those to randomly omit hidden units - very 
		powerful technique'''

		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_
		
		self.act[0] = self.activ[0](T.dot(X,wts[0]) + bs[0]) # use the first data matrix to compute the first activation
		if len(wts) > 1: # len(wts) = 1 corresponds to softmax regression
			for i,(w,b,activ) in enumerate(zip(wts[1:],bs[1:],self.activ[1:])):
				self.act[i+1] = activ(T.dot(self.act[i],w) + b)

	# compute_loss and compute_grad functions - the former applies forward propagation, and computes the loss
	# based on the labels. The latter applies autodiff on this loss function and computes the gradients with 
	# respect to the weights and biases

	def compute_loss(self,X,y,wts=None,bs=None):
		''' Given inputs (X,y), returns the loss at the current state of the model (wts,bs) '''
		
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		self.fprop(X,wts,bs)
		loss = self.loss_func(y)
		
		return loss

	def compute_grad(self,loss,wts=None,bs=None):
		''' Given the loss, computes its derivative with respect to the weights and biases of the
		neural network '''

		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		d_loss_d_params = T.grad(loss,[p for param in [wts,bs] for p in param])
		d_loss_d_wts = d_loss_d_params[:len(wts)]
		d_loss_d_bs = d_loss_d_params[len(wts):]

		return d_loss_d_wts,d_loss_d_bs

	# A few basic loss functions that are used often. The autoencoder builds upon this and adds
	# a sparsity constraint with a few more loss parameters. In general though, one can still 
	# define custom loss functions and feed those into this base class 
	
	def regularization_loss(self):
		''' L1 or L2 regularization '''
		reg_loss = 0

		if 'L1_decay' in self.loss_params:
			reg_loss += self.loss_params['L1_decay']*sum([T.sum(T.abs_(w)) for w in self.wts_])
		
		if 'L2_decay' in self.loss_params:
			reg_loss += 0.5*self.loss_params['L2_decay']*sum([T.sum(w**2) for w in self.wts_])

		return reg_loss

	def cross_entropy(self,y):
		''' basic cross entropy loss function with optional regularization'''
		
		E = T.mean(T.sum(-1.0*y*T.log(self.act[-1]),axis=1)) + self.regularization_loss()

		return E

	def squared_error(self,y):
		''' basic squared error loss function with optional regularization'''

		E = T.mean(T.sum((y-self.act[-1])**2)) + self.regularization_loss()

		return E