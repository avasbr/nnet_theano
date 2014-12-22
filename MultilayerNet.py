import numpy as np
import NeuralNetworkCore
import theano
import theano.tensor as T
import Autoencoder as ae
import sys
from theano.tensor.shared_randomstreams import RandomStreams

class MultilayerNet(NeuralNetworkCore.Network):

	def __init__(self,d=None,k=None,n_hid=None,activ=None,dropout_flag=False,input_p=None,
		hidden_p=None, loss_type='cross_entropy',**loss_params):
		''' simply calls the superclass constructor with the appropriate loss function'''
		
		self.dropout_flag = dropout_flag
		
		# add defensive coding here
		self.input_p = input_p
		self.hidden_p = hidden_p
		self.srng = RandomStreams() # initialize the random number stream

		if loss_type == 'cross_entropy':
			super(MultilayerNet,self).__init__(d=d,k=k,n_hid=n_hid,activ=activ,loss_func=self.cross_entropy,**loss_params)		
		
		elif loss_type == 'squared_error':
			super(MultilayerNet,self).__init__(d=d,k=k,n_hid=n_hid,activ=activ,loss_func=self.squared_error,**loss_params)
		
		else:
			sys.exit("That loss function is not available")

		# functions which will be compiled after 'fit' has been run
		self.predict = None
		self.score = None 

	def dropout(self,act,p=0.):
		''' Randomly drops an activation with probability p '''
		if p > 0:
			# randomly dropout p activations 
			retain_prob = 1-p
			return act*self.srng.binomial(act.shape,p=retain_prob,dtype=theano.config.floatX)

	def dropout_fprop(self,X,wts=None,bs=None):
		''' Performs forward propagation through the network, incorporating dropout, 
		and updates all intermediate values '''
		
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		self.act[0] = self.activ[0](T.dot(self.dropout(X,self.input_p),wts[0]) + bs[0]) # use the first data matrix to compute the first activation
		if len(wts) > 1: # len(wts) = 1 corresponds to softmax regression
			for i,(w,b,activ) in enumerate(zip(wts[1:],bs[1:],self.activ[1:])):
				self.act[i+1] = activ(T.dot(self.dropout(self.act[i],self.hidden_p),w) + b)

	def fit(self,X,y,wts=None,bs=None,X_val=None,y_val=None,**optim_params):
		''' calls the fit function of the super class (NeuralNetworkCore) and also compiles the 
		prediction and scoring functions '''
		
		super(MultilayerNet,self).fit(X,y,wts,bs,**optim_params)
		self.compile_multilayer_functions(wts,bs)

	def compute_loss(self,X,y,wts=None,bs=None):
		''' Given inputs (X,y), returns the loss at the current state of the model (wts,bs) '''
		
		# based on if dropout needs to be used, this function will either 
		if self.dropout_flag:
			if wts is None and bs is None:
				wts = self.wts_
				bs = self.bs_

			# the optimization loss is based on the output from applying dropout
			self.dropout_fprop(X,wts,bs)
			optim_loss = self.loss_func(y) + self.regularization(wts)

			# the evaluation loss is based on the expected value of the weights 
			eval_wts = [wt.get_value() for wt in self.wts_]
			eval_bs = [b.get_value() for b in self.bs_]
			eval_wts[0] *= self.input_p
			for wt in eval_wts[1:]:
				wt *= self.hidden_p
			self.fprop(X,eval_wts,eval_bs)
			eval_loss = self.loss_func(y)

			return optim_loss,eval_loss

		else:
			return super(MultilayerNet,self).compute_loss(X,y,wts,bs)

	def compile_multilayer_functions(self,wts=None,bs=None):
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
		self.predict = theano.function([X],pred,mode='FAST_RUN',allow_input_downcast=True)
		self.score = theano.function([X,y],1.0-T.mean(T.neq(pred,T.argmax(y,axis=1))),mode='FAST_RUN',allow_input_downcast=True)	