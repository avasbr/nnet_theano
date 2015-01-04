import numpy as np
import NeuralNetworkCore
import theano
import theano.tensor as T
import sys

class Autoencoder(NeuralNetworkCore.Network):

	def __init__(self,d=None,num_hid=None,activ=None,loss_terms=None,**loss_params):
		''' implementation of the basic autoencoder '''
		
		# the autoencoder can only have one hidden layer (and therefore, only two activation functions)
		assert isinstance(n_hid,int) and len(activ) == 2 

		super(Autoencoder,self).__init__(d=d,k=k,num_hid=[num_hid],activ=activ,loss_terms=loss_terms,**loss_params)

		# functions that will be available after running the 'fit' method on the autoencoder
		self.decode = None
		self.encode = None
		self.get_pretrained_weights = None

	def gauss_corrupt_input(X,sigma=0.1):
		''' adds gaussian noise to the input, making this autoencoder a 'denoising' flavor'''

	def binary_corrupt_input(X,p=0.05):
		''' similar to the gaussian case, but for binary inputs; this simply flips the values
		of p*d random dimensions per feature vector''' 

	def fit(self,X_tr,wts=None,bs=None,X_val=None,**optim_params):
		''' calls the fit function of the super class (NeuralNetworkCore) and also compiles the 
		encoding and decoding functions'''
		
		super(Autoencoder,self).fit(X_tr,X_tr,X_val=X_val,y_val=X_val,**optim_params)
		self.compile_autoencoder_functions()		

	def fprop(self,X_tr,wts=None,bs=None):
		''' Performs forward propagation through the network - this fprop is simplified
		specifically for autoencoders, which have only one hidden layer

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

		# this is useful to keep around, if we introduce sparsity
		self.hidden_act = self.activ[0](T.dot(X_tr,wts[0]) + bs[0]) 
		return activ[1](T.dot(hidden_act,wts[1]) + bs[1])

	def compute_loss(self,X,y,wts=None,bs=None):
		''' Given inputs, returns the loss at the current state of the model'''
		
		# call the super-class function first...		
		optim_loss, eval_loss = super(Autoencoder,self).compute_loss(X,y,wts,bs)

		# ... and augment with the sparsity term, if needed
		if 'sparsity' in self.loss_terms:
			beta = self.loss_params.get('beta')
			rho = self.loss_params.get('rho')
			optim_loss += nl.sparsity(self.hidden_act,beta=beta,rho=rho) 

		# no changes to the eval_loss function
		return optim_loss,eval_loss

	def compile_autoencoder_functions(self,wts=None,bs=None):
		''' compiles the encoding, decoding, and pre-training functions of the autoencoder 
		
		Parameters
		----------
		param: wts - weights
		type: theano matrix

		param: bs - biases
		type: theano matrix
		'''
		
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		X = T.matrix() # features to encode or decode

		self.encode = theano.function(inputs=[X],outputs=self.activ[0](T.dot(X,wts[0])+b[0]))
		self.decode = theano.function(inputs=[X],outputs=self.activ[1](T.dot(X,wts[1])+b[1]))
		self.get_pretrained_weights = theano.function(inputs=[X],outputs=[wts[0].get_value(),bs[0].get_value()])