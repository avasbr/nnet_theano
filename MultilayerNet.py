import numpy as np
import NeuralNetworkCore
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T
import theano
import Autoencoder as ae
import nnetutils as nu
import sys

class MultilayerNet(NeuralNetworkCore.Network):

	def __init__(self,d=None,k=None,num_hid=None,activ=None,dropout_flag=False,input_p=None,
		hidden_p=None, loss_func=nu.cross_entropy,**loss_params):
		''' simply calls the superclass constructor with the appropriate loss function'''
		
		self.dropout_flag = dropout_flag
		
		# add defensive coding here
		self.input_p = input_p
		self.hidden_p = hidden_p
		self.srng = RandomStreams() # initialize the random number stream

		super(MultilayerNet,self).__init__(d=d,k=k,num_hid=num_hid,activ=activ,loss_func=loss_func,**loss_params)
		
		# functions which will be compiled after 'fit' has been run
		self.predict = None
		self.score = None 

	def dropout(self,act,p=0):
		''' Randomly drops an activation with probability p '''
		if p > 0:
			# randomly dropout p activations 
			retain_prob = 1.-p
			return (1./retain_prob)*act*self.srng.binomial(act.shape,p=retain_prob,dtype=theano.config.floatX)

	def dropout_fprop(self,X,wts=None,bs=None):
		''' forward propagation with dropout, for training '''
		
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		act = self.activ[0](T.dot(self.dropout(X,self.input_p),wts[0]) + bs[0]) # compute the first activation
		if len(wts) > 1: # len(wts) = 1 corresponds to softmax regression
			for i,(w,b,activ) in enumerate(zip(wts[1:],bs[1:],self.activ[1:])):
				act = activ(T.dot(self.dropout(act,self.hidden_p),w) + b)

		return act

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
			y_prob = self.dropout_fprop(X,wts,bs)
			optim_loss = self.loss_func(y,y_prob) + nu.regularization(wts)

			# the evaluation loss is based on the expected value of the weights 
			# y_prob = self.dropout_eval_fprop(X,wts,bs)
			y_prob = self.fprop(X,wts,bs)
			eval_loss = self.loss_func(y,y_prob)

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

		# if dropout was used, we have to rescale everything, otherwise, just go back to the default
		y_prob = self.fprop(X,wts,bs)
		# if self.dropout_flag:
		# 	y_prob = self.dropout_eval_fprop(X,wts,bs)
		# else:
		# 	y_prob = self.fprop(X,wts,bs)
		
		pred = T.argmax(y_prob,axis=1)

		# compile the functions - this is what the user can use to do prediction
		self.predict = theano.function([X],pred,mode='FAST_RUN',allow_input_downcast=True)
		self.score = theano.function([X,y],1.0-T.mean(T.neq(pred,T.argmax(y,axis=1))),mode='FAST_RUN',allow_input_downcast=True)	