import numpy as np
from deepnet import NeuralNetworkCore
import theano.tensor as T
import theano
from deepnet import Autoencoder as ae
from deepnet.common import nnetutils as nu
import sys

class MultilayerNet(NeuralNetworkCore.Network):
	''' The classic, multilayer neural network, with prediciton and scoring functions '''

	def __init__(self,d=None,k=None,num_hids=None,activs=None,loss_terms=None,**loss_params):
		''' simply calls the superclass constructor with the appropriate loss function'''
		
		super(MultilayerNet,self).__init__(d=d,k=k,num_hids=num_hids,activs=activs,loss_terms=loss_terms,**loss_params)
		
		# functions which will be compiled after 'fit' has been run
		self.predict = None
		self.score = None 

	def fit(self,X_tr,y_tr,X_val=None,y_val=None,wts=None,bs=None,**optim_params):
		''' calls the fit function of the super class (NeuralNetworkCore) and also compiles the 
		prediction and scoring functions 

		Parameters:
		-----------
		param: X_tr - training data
		type: theano matrix

		param: y_tr - training labels
		type: theano matrix

		param: X_val - validation data
		type: theano matrix

		param: y_val - validation labels
		type: theano matrix

		param: **optim_params - optimization parameters
		type: dictionary

		'''
		super(MultilayerNet,self).fit(X_tr,y_tr,X_val=X_val,y_val=y_val,wts=wts,bs=bs,**optim_params)
		self.compile_multilayer_functions()

	def compile_multilayer_functions(self,wts=None,bs=None):
		''' compiles prediction and scoring functions for testing 

		Parameters:
		-----------
		param: wts - weights
		type: numpy ndarray matrix

		param: bs - biases
		type: numpy ndarray matrix
		'''

		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_
		else:
			wts = [nu.floatX(w) for w in wts]
			bs = [nu.floatX(b) for b in bs]

		X = T.matrix()
		y = T.matrix()

		y_prob = self.fprop(X,wts,bs) # forward propagate through the network
		pred = T.argmax(y_prob,axis=1) # predict outputs

		# compile the functions - this is what the user can use to do prediction and scoring
		self.predict = theano.function(
			inputs=[X],
			outputs=pred,
			mode='FAST_RUN',
			allow_input_downcast=True)
		
		self.score = theano.function(
			inputs=[X,y],
			outputs=1.0-T.mean(T.neq(pred,T.argmax(y,axis=1))),
			mode='FAST_RUN',
			allow_input_downcast=True)