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

def reLU(z):
	''' rectified linear activation function '''
	return T.maximum(0,z)

def split_train_val(X,y,p):
	''' splits into training and validation sets '''
	
	m = X.shape[0]
	idx = np.random.permutation(m)
	m_tr = m*p
	tr_idx = idx[:m_tr]
	val_idx = idx[m_tr:]
	X_tr = X[tr_idx]
	X_val = X[val_idx]
	y_tr = y[tr_idx]
	y_val = y[val_idx]

	return X_tr,y_tr,X_val,y_val
