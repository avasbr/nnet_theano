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

def theano_unroll(wts,bs):
	'''Flattens matrices and concatenates to a vector '''
	v = floatX(np.array([]))
	for w,b in zip(wts,bs):
		v = T.concatenate((v,T.flatten(w),T.flatten(b)))
	return v

def theano_reroll(v,n_nodes):
	'''Re-rolls a vector v into the weight matrices'''

	idx = 0
	r_wts = []
	r_bs = []
	for row,col in zip(n_nodes[:-1],n_nodes[1:]):
		w_size = row*col; b_size = col
		r_wts.append(T.reshape(v[idx:idx+w_size],(row,col))); idx += w_size
		r_bs.append(T.reshape(v[idx:idx+b_size],(col,))); idx += b_size
	
	return r_wts,r_bs

def unroll(wts,bs):
	'''Flattens matrices and concatenates to a vector '''
	v = np.array([])
	for w,b in zip(wts,bs):
		v = np.concatenate((v,w.flatten(),b.flatten()))
	return v

def reroll(v,n_nodes):
	'''Re-rolls a vector v into the weight matrices'''

	idx = 0
	r_wts = []
	r_bs = []
	for row,col in zip(n_nodes[:-1],n_nodes[1:]):
		w_size = row*col; b_size = col
		r_wts.append(np.reshape(v[idx:idx+w_size],(row,col))); idx += w_size
		r_bs.append(np.reshape(v[idx:idx+b_size],(col,))); idx += b_size
	
	return r_wts,r_bs
