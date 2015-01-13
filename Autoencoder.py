import numpy as np
import NeuralNetworkCore
import theano
import theano.tensor as T
import sys

class Autoencoder(NeuralNetworkCore.Network):

	def __init__(self,d=None,num_hid=None,activs=None,tied_wts=False,loss_terms=None,**loss_params):
		''' implementation of the basic autoencoder '''
		
		# the autoencoder can only have one hidden layer (and therefore, only two activation functions)
		assert isinstance(n_hid,int) and len(activs) == 2 

		super(Autoencoder,self).__init__(d=d,k=k,num_hids=[num_hid],activs=activs,loss_terms=loss_terms,**loss_params)

		# functions that will be available after running the 'fit' method on the autoencoder
		self.decode = None
		self.encode = None
		self.get_pretrained_weights = None
		self.tied_wts = tied_wts

	def set_weights(self,wts=None,bs=None,method='gauss'):
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
		# with tied weights, the encoding and decoding matrices are simply transposes
		# of one another

		if self.tied_wts:
			self.num_nodes = [d]+num_hids
			
			# weights and biases
			if wts is None and bs is None:
				self.wts_ = [None,None]
				self.bs_ = [None,None]

				if method == 'gauss':
					self.wts_[0] = theano.shared(nu.floatX(0.01*np.random.randn(d,num_hids[0])))
					self.wts_[1] = self.wts_[0].T # this is a shallow transpose (changing [0] will change this as well)
					self.bs_[0] = theano.shared(nu.floatX(np.zeros(num_hids[0])))
					self.bs_[1] = theano.shared(nu.floatX(np.zeros(d)))

				if method == 'random':
					v = np.sqrt(1./(d+num_hids[0]+1))
					self.wts_[0] = theano.shared(nu.floatX(2.0*v*np.random.rand(d,num_hids[0]-v)))
					self.wts_[1] = self.wts_]0].T
					self.bs_[0] = theano.shared(nu.floatX(np.zeros(num_hids[0])))
					self.bs_[1] = theano.shared(nu.floatX(np.zeros(d)))

		# if encoding and decoding matrices are distinct, just default back to the normal case
		else:
			super(Autoencoder,self).set_weights(wts,bs,method)

	def corrupt_input(X,v=0.1,method='mask'):
		''' corrupts the input using one of several methods

		Parameters:
		-----------
		param: X - input matrix
		type: np.ndarray

		param: v - either the proportion of values to corrupt, or std for gaussian
		type: float

		param: method - either 'mask', 'gauss'
		type: string
		'''
		if method == 'mask':
			return X*self.srng.binomial(X.shape,n=1,p=1-v,dtype=theano.config.floatX))
		elif method == 'gauss':
			return X*self.srng.normal(X.shape,avg=0.0,std=v,dtype=theano.config.floatX)
		# TODO: will probably want to add support for "salt-and-pepper" (essentially an XOR)
		# noise. See the theano notes for the implementation, though to use bitwise ops, everything
		# needs to be an int...

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