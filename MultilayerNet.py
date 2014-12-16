import numpy as np
import NeuralNetworkCore
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class MultilayerNet(NeuralNetworkCore.Network):

	def __init__(self,d=None,k=None,n_hid=None,activ=None,dropout_flag=False,input_p=None,
		hidden_p=None, cost_type='cross_entropy',**cost_params):
		''' simply calls the superclass constructor with the appropriate cost function'''
		
		self.dropout_flag = dropout_flag
		# add defensive coding here
		self.input_p = input_p
		self.hidden_p = hidden_p
		self.srng = RandomStreams() # initialize the random number stream

		if cost_type == 'cross_entropy':
			super(MultilayerNet,self).__init__(d=d,k=k,n_hid=n_hid,activ=activ,cost_func=self.cross_entropy,**cost_params)		
		
		elif cost_type == 'squared_error':
			super(MultilayerNet,self).__init__(d=d,k=k,n_hid=n_hid,activ=activ,cost_func=self.squared_error,**cost_params)
		
		else:
			sys.exit("That cost function is not available")

	def dropout(self,act,p=0.):
		''' Randomly drops an activation with probability p '''
		if p > 0:
			# randomly dropout p activations 
			retain_prob = 1-p
			act *= self.srng.binomial(act.shape,p=retain_prob,dtype=theano.config.floatX)

	def fprop(self,X,wts=None,bs=None):
		''' Performs forward propagation through the network, and updates all intermediate values. if
		input_dropout and hidden_dropout are specified, uses those to randomly omit hidden units - very 
		powerful technique'''
		
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		# if we're doing dropout...
		if self.dropout_flag:
			self.dropout(X,self.input_p) # apply dropout to the input
			self.act[0] = self.activ[0](T.dot(X,wts[0]) + bs[0]) # use the first data matrix to compute the first activation
			if len(wts) > 1: # len(wts) = 1 corresponds to softmax regression
				for i,(w,b,activ) in enumerate(zip(wts[1:],bs[1:],self.activ[1:])):
					self.dropout(self.act[i],self.hidden_p) # apply dropout to the hidden units
					self.act[i+1] = activ(T.dot(self.act[i],w) + b)

		# otherwise, just fall back to normal forward propagation in the superclass
		else:
			super(MultilayerNet,self).fprop(X,wts,bs)

	def fit(self,X,y,wts=None,bs=None,X_val=None,y_val=None,**optim_params):
		''' calls the fit function of the super class (NeuralNetworkCore) and also compiles the 
		prediction and scoring functions '''
		
		super(MultilayerNet,self).fit(X,y,wts,bs,**optim_params)
		self.compile_predict_score(wts,bs)

	def compile_predict_score(self,wts=None,bs=None):
		''' compiles prediction and scoring functions for testing '''

		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		X = T.matrix()
		y = T.matrix()

		# if dropout was used, we have to rescale everything
		if self.dropout_flag:
			wts[0] *= self.input_p
			for w in wts[1:]:
				w *= self.hidden_p

		self.fprop(X,wts,bs)
		pred = T.argmax(self.act[-1],axis=1)

		# compile the functions - this is what the user can use to do prediction
		self.predict = theano.function([X],pred)
		self.score = theano.function([X,y],1.0-T.mean(T.neq(pred,T.argmax(y,axis=1))))	