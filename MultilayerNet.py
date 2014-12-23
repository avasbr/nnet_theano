import numpy as np
import NeuralNetworkCore
import theano.tensor as T
import theano
import Autoencoder as ae
import nnetutils as nu
import sys

class MultilayerNet(NeuralNetworkCore.Network):

	def __init__(self,d=None,k=None,num_hid=None,activ=None,loss_func=nu.cross_entropy,**loss_params):
		''' simply calls the superclass constructor with the appropriate loss function'''
		
		super(MultilayerNet,self).__init__(d=d,k=k,num_hid=num_hid,activ=activ,loss_func=loss_func,**loss_params)
		
		# functions which will be compiled after 'fit' has been run
		self.predict = None
		self.score = None 

	def fit(self,X,y,wts=None,bs=None,X_val=None,y_val=None,**optim_params):
		''' calls the fit function of the super class (NeuralNetworkCore) and also compiles the 
		prediction and scoring functions '''
		
		super(MultilayerNet,self).fit(X,y,wts,bs,**optim_params)
		self.compile_multilayer_functions(wts,bs)


	def compile_multilayer_functions(self,wts=None,bs=None):
		''' compiles prediction and scoring functions for testing '''

		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		X = T.matrix()
		y = T.matrix()

		# if dropout was used, we have to rescale everything, otherwise, just go back to the default
		y_prob = self.fprop(X,wts,bs)

		pred = T.argmax(y_prob,axis=1)

		# compile the functions - this is what the user can use to do prediction
		self.predict = theano.function([X],pred,mode='FAST_RUN',allow_input_downcast=True)
		self.score = theano.function([X,y],1.0-T.mean(T.neq(pred,T.argmax(y,axis=1))),mode='FAST_RUN',allow_input_downcast=True)	