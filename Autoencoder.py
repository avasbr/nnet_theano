import numpy as np
import NeuralNetworkCore
import theano
import theano.tensor as T
import sys

class Autoencoder(NeuralNetworkCore.Network):

	def __init__(self,d=None,n_hid=None,activ=None,loss_type='cross_entropy',**loss_params):
		''' simply calls the superclass constructor with the appropriate loss function'''
		
		# the autoencoder can only have one hidden layer (and therefore, only two activation functions)
		assert isinstance(n_hid,int) and len(activ) == 2 

		if loss_type == 'cross_entropy':
			NeuralNetworkCore.Network.__init__(self,d=d,k=d,n_hid=[n_hid],activ=activ,loss_func=self.sparse_cross_entropy,**loss_params)
		
		elif loss_type == 'squared_error':
			NeuralNetworkCore.Network.__init__(self,d=d,k=d,n_hid=[n_hid],activ=activ,loss_func=self.sparse_squared_error,**loss_params)
		
		else:
			sys.exit("That loss function is not available")

		self.decode = None
		self.encode = None

	def sparsity_loss(self,wts=None,bs=None):
		
		sparse_loss = 0

		if 'beta' in self.loss_params and 'rho' in self.loss_params:
			beta = self.loss_params['beta']
			rho = self.loss_params['rho']
 			avg_act = T.mean(self.act[0],axis=0)

 			sparse_loss = beta*T.sum(rho*T.log(rho/avg_act)+(1-rho)*T.log((1-rho)/(1-avg_act)))
		
		return sparse_loss

	def sparse_cross_entropy(self,y,wts=None,bs=None):
		''' cross entropy with a sparsity constraint '''

		return self.cross_entropy(y,wts) + self.sparsity_loss(wts)

	def sparse_squared_error(self,y,wts=None,bs=None):
		''' squared error with a sparsity constraint '''

		return self.squared_error(y,wts) + self.sparsity_loss(wts)

	def fit(self,X,y,wts=None,bs=None,X_val=None,y_val=None,**optim_params):
		''' calls the fit function of the super class (NeuralNetworkCore) and also compiles the 
		encoding and decoding functions '''
		
		super(MultilayerNet,self).fit(X,y,wts,bs,**optim_params)
		self.compile_predict_score(wts,bs)

	def compile_encode_decode(self,wts=None,bs=None):
		''' compiles the encoding and decoding functions of the autoencoder '''
		
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		X = T.matrix() # features to encode or decode

		self.encode = theano.function(inputs=[X],outputs=self.activ[0](T.dot(X,wts[0])+b[0]))
		self.decode = theano.function(inputs=[X],outputs=self.activ[1](T.dot(X,wts[1])+b[1]))

	def pretrain(self,X_in,wts=None,bs=None,**optim_params):
		''' convenience function which calls 'fit' and returns the encoding
		weight matrix, which would be used for initializing weights prior to
		doing fine-tuned supervised training '''
		
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		self.fit(X_in,X_in,wts,bs,**optim_params)
		
		return wts[0],bs[0]