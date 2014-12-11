import numpy as np
import NeuralNetworkCore
import theano
import theano.tensor as T

class Autoencoder(NeuralNetworkCore.Network):

	def __init__(self,d=None,k=None,n_hid=None,activ=None,cost_type='cross_entropy',**cost_params):
		''' simply calls the superclass constructor with the appropriate cost function'''
		
		# the autoencoder can only have one hidden layer (and therefore, only two activation functions)
		assert len(n_hid) == 1 and len(activ) == 2

		if cost_type == 'cross_entropy':
			NeuralNetworkCore.Network.__init__(self,d=d,k=k,n_hid=n_hid,activ=activ,cost_func=self.sparse_cross_entropy,**cost_params)
		
		elif cost_type == 'squared_error':
			NeuralNetworkCore.Network.__init__(self,d=d,k=k,n_hid=[n_hid],activ=activ,cost_func=self.sparse_squared_error,**cost_params)
		
		else:
			sys.exit("That cost function is not available")

	def sparsity_cost(self,wts=None,bs=None,**cost_params):
		
		beta = cost_params['beta']
		rho = cost_params['rho']
 		avg_act = T.mean(self.act[0],axis=1)

		return beta*T.sum(rho*T.log(rho/avg_act)+(1-rho)*T.log((1-rho)/(1-avg_act)))

	def sparse_cross_entropy(self,wts=None,bs=None,**cost_params):
		''' cross entropy with a sparsity constraint '''

		return self.cross_entropy(wts,**cost_params) + self.sparsity_cost(wts,**cost_params)

	def sparse_squared_error(self,wts=None,bs=None,**cost_params):
		''' squared error with a sparsity constraint '''

		return self.squared_error(wts,**cost_params) + self.sparsity_cost(wts,**cost_params)

	def decode(self,X_e,wts=None,bs=None):
		''' takes encoded feature vectors and decodes them ''' 
		
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		return self.activ[1](T.dot(X_e,wts[1]) + bs[1])

	def encode(self,X,wts=None,bs=None):
		''' encodes the original features '''
		
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		return self.activ[0](T.dot(X,wts[0]) + bs[0])

	def pretrain(self,X_in,wts=None,bs=None,**optim_params):
		''' convenience function which calls 'fit' and returns the encoding
		weight matrix, which would be used for initializing weights prior to
		doing fine-tuned supervised training '''
		
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		self.fit(X_in,X_in,wts,bs,**optim_params)
		
		return wts[0]