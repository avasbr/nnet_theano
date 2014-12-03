import theano
import theano.tensor as T
import numpy as np

def floatX(X):
	return np.asarray(X,dtype=theano.config.floatX)

def sigmoid(z):
	''' sigmoid activation function '''
	return 1./(1.+T.exp(-1.*z))

def softmax(z):
	''' softmax activation function '''
	max_v = T.max(z,axis=0,keepdims=True)
	log_sum = T.log(T.sum(T.exp(z-max_v),axis=0)) + max_v
	return T.exp(z-log_sum)

def rectilinear(z):
	''' rectified linear activation function '''
	return T.maximum(0,z)