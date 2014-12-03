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

	def __init__(self,d=None,k=None,n_hid=None,activ=None,cost_func=None,**cost_params):

		# network parameters
		self.n_nodes = [d]+n_hid+[k] # number of nodes
		self.act = (len(self.n_nodes)-1)*[None]
		self.activ = activ
		
		if all(node for node in self.n_nodes):
			self.set_weights()
		
		self.cost_func = cost_func
		self.cost_params = cost_params

	def set_weights(self,wts=None,bs=None):
		''' Initializes the weights and biases of the neural network '''

		# weights and biases
		if wts is None and bs is None:
			self.wts_ = (len(self.n_nodes)-1)*[None]
			self.bs_ = (len(self.n_nodes)-1)*[None]
			
			for i,(n1,n2) in enumerate(zip(self.n_nodes[:-1],self.n_nodes[1:])):
				v = 4*np.sqrt(6./(n1+n2))
				self.wts_[i] = theano.shared(nu.floatX(2.0*v*np.random.rand(n2,n1)-v)) # convert to the floatX format
				self.bs_[i] = theano.shared(nu.floatX(np.zeros((n2,1))))
		else:
			assert isinstance(wts,list)
			assert isinstance(bs,list)
			self.wts_ = [theano.shared(nu.floatX(w)) for w in wts]
			self.bs_ = [theano.shared(nu.floatX(b)) for b in bs]

	def fit(self,X,y,wts=None,bs=None,**optim_params):
		''' Short description
		
		Parameters:
		-----------
		
		Returns:
		--------
		'''
		def method_err():
			err_msg = ('No method provided to fit! Your choices are:'
						'\n(1) SGD: stochastic gradient descent'+
						'\n(2) SGDm: stochastic gradient descent with momentum'
						'\n(3) SGDim: an improved version of SGDm'
						'\n(4) RMSPROP: hintons mini-batch mini-batch version of rprop [UNDER CONSTRUCTION]')
			return err_msg

		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		if 'method' not in optim_params or optim_params['method'] is None:
			sys.exit(method_err())
		
		method = optim_params['method']
		del optim_params['method']

		if method == 'SGD':
			wts,bs = nopt.gradient_descent(X,y,wts,bs,self.compute_cost,**optim_params)

		elif method == 'SGDm':
			pass
			
		elif method == 'SGDim':
			pass
		else:
			print method_err()

		return self

	def fprop(self,X,wts=None,bs=None):
		''' Performs forward propagation through the network, and updates all intermediate values'''
		
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		self.act[0] = self.activ[0](T.dot(wts[0],X) + bs[0]) # use the first data matrix to compute the first activation
		if len(wts) > 1: # len(wts) = 1 corresponds to softmax regression
			for i,(w,b,activ) in enumerate(zip(wts[1:],bs[1:],self.activ[1:])):
				self.act[i+1] = activ(T.dot(w,self.act[i]) + b)


	# This might seem redundant and it's mostly for convenience, but there's a reason why it makes sense to break these into
	# two separate functions. "compute_cost_grad" lumps the computation of the cost function and gradient function into one,
	# and is particularly useful for optimization over a training set. We get the cost for free, since we need to compute the
	# cost en route to computing the gradient anyway. "compute_cost", as the name suggests, only computes the cost, and this is
	# useful for tracking performance on validation and test sets. In general, both of these functions would be used in a 
	# gradient-based optimizer (e.g. sgd, adagrad, rmsprop, etc) 

	def compute_cost(self,X,y,wts=None,bs=None):
		''' Given inputs (X,y), returns the cost at the current state of the model (wts,bs) '''

		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		self.fprop(X,wts,bs)
		E = self.cost_func(y,wts,bs,**self.cost_params)
		
		return E

	# def compute_cost_grad(self,X,y,wts=None,bs=None):
	# 	''' Given inputs (X,y), returns the cost at the current state of the model (wts,bs), as well as the gradient of '''

	# 	if wts is None and bs is None:
	# 		wts = self.wts_
	# 		bs = self.bs_

	# 	E = self.compute_cost(X,y,wts,bs)
	# 	grads = T.grad(E, [p for sublist in [wts,bs] for p in sublist]) # auto-diff (implements backprop)
	# 	dW = grads[:len(wts)] # collect gradients for weight matrices...
	# 	db = grads[len(wts):]# ...and biases

	# 	return E,dW,db # return the cost and gradients

