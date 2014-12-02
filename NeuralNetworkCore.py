import cPickle
import os
import sys
import time
import numpy as np
import theano
import theano.tensor as T

class Network(object):

	def __init__(self,d=None,k=None,n_hid=None,activ=None,cost_func=None,**hyperparam):

		# network parameters
		self.n_nodes = [d]+n_hid+[k] # number of nodes
		self.act = (len(self.n_nodes)-1)*[None]
		self.activ = activ
		if all(node for node in self.n_nodes):
			self.set_weights(method='alt_random')
		self.cost_func = cost_func
		self.hyperparam = hyperparam

	def floatX(X):
		return np.asarray(X,dtype=theano.config.floatX)

	def set_weights(self,wts=None,bs=None):
		''' Initializes the weights and biases of the neural network '''

		if wts is None and bs is None:
			for i,(n1,n2) in enumerate(zip(self.n_nodes[:-1],self.n_nodes[1:])):
				v = 4*np.sqrt(6./(n1+n2))
				self.wts_[i] = theano.shared(floatX(2.0*v*np.random.rand(n2,n1)-v)) # convert to the floatX format
				self.bs_[i] = theano.shared(floatX(np.zeros(n2,1)))
		else:
			assert isinstance(wts,list)
			assert isinstance(bs,list)
			self.wts_ = [theano.shared(floatX(w)) for w in wts]
			self.bs_ = [theano.shared(floatX(b)) for b in bs]

	def fprop(self,X,y,wts=None,bs=None):
		''' Performs forward propagation through the network, and updates all intermediate values'''
		
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		self.act[0] = self.activ[0](T.dot(wts[0],X) + bs[0]) # use the first data matrix to compute the first activation
		if len(wts) > 1: # len(wts) = 1 corresponds to softmax regression
			for i,(w,b,activ) in enumerate(zip(wts[1:],bs[1:],self.activ[1:])):
				self.act[i+1] = activ(T.dot(w,self.act[i]) + b)

	def compute_cost_grad(self,X,y,wts=None,bs=None):
		''' Given inputs (X,y), returns the cost at the current state of the model (wts,bs), as well as the gradient of '''

		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_
		self.fprop(X,y,wts,bs) # forward propagation
		E = self.cost_func(X,y,wts,bs,**self.hyperparam) # cost function
		grads = T.grad(E, [p for sublist in [wts,bs] for p in sublist]) # auto-diff (implements backprop)
		dW = grads[:len(wts)] # collect gradients for weight matrices...
		db = grads[len(wts):]# ...and biases

		return E,dW,db