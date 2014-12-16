import numpy as np
import NeuralNetworkCore
import theano
import theano.tensor as T
import sys

class Autoencoder(NeuralNetworkCore.Network):

	def __init__(self,d=None,n_hid=None,activ=None,cost_type='cross_entropy',**cost_params):
		''' simply calls the superclass constructor with the appropriate cost function'''
		
		# the autoencoder can only have one hidden layer (and therefore, only two activation functions)
		assert isinstance(n_hid,int) and len(activ) == 2 

		if cost_type == 'cross_entropy':
			NeuralNetworkCore.Network.__init__(self,d=d,k=d,n_hid=[n_hid],activ=activ,cost_func=self.sparse_cross_entropy,**cost_params)
		
		elif cost_type == 'squared_error':
			NeuralNetworkCore.Network.__init__(self,d=d,k=d,n_hid=[n_hid],activ=activ,cost_func=self.sparse_squared_error,**cost_params)
		
		else:
			sys.exit("That cost function is not available")

		self.decode = None
		self.encode = None

	def sparsity_cost(self,wts=None,bs=None):
		
		sparse_cost = 0

		if 'beta' in self.cost_params and 'rho' in self.cost_params:
			beta = self.cost_params['beta']
			rho = self.cost_params['rho']
 			avg_act = T.mean(self.act[0],axis=0)

 			sparse_cost = beta*T.sum(rho*T.log(rho/avg_act)+(1-rho)*T.log((1-rho)/(1-avg_act)))
		
		return sparse_cost

	def sparse_cross_entropy(self,y,wts=None,bs=None):
		''' cross entropy with a sparsity constraint '''

		return self.cross_entropy(y,wts) + self.sparsity_cost(wts)

	def sparse_squared_error(self,y,wts=None,bs=None):
		''' squared error with a sparsity constraint '''

		return self.squared_error(y,wts) + self.sparsity_cost(wts)

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
		
		return wts[0]