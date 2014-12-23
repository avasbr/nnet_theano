import theano
import theano.tensor as T
import numpy as np

def floatX(X):
	return np.asarray(X,dtype=theano.config.floatX)

def sigmoid(z):
	''' sigmoid activation function '''
	return 1./(1.+T.exp(-1.*z))

def tanh(z):
	''' hyperbolic tangent activation function '''
	c = T.exp(z)
	c_ = T.exp(-1.*z)
	return (c - c_)/(c + c_)

def softmax(z):
	''' softmax activation function '''
	max_v = T.max(z,axis=0,keepdims=True)
	log_sum = T.log(T.sum(T.exp(z-max_v),axis=0)) + max_v
	return T.exp(z-log_sum)

def reLU(z):
	''' rectified linear activation function '''
	return 0.5*(z + abs(z))
	# return T.maximum(z,0)

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

# Ideally there should just be a single set of functions for this, instead of splitting them up like this - but for
# the time being, this works, and since this is primarily for gradient checking (not a super-critical piece of this
# project, given that it's theano's autodiff, which has presumably been tested to death), i'm not going to focus a lot
# of time to clean this up at the moment
def t_unroll(wts,bs):
	'''Flattens matrices and concatenates to a vector - need for constructing theano expression graphs'''
	v = floatX(np.array([]))
	for w,b in zip(wts,bs):
		v = T.concatenate((v,T.flatten(w),T.flatten(b)))
	return v

def t_reroll(v,n_nodes):
	'''Re-rolls a vector v into the weight matrices - need for constructing theano expression graphs'''

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

def regularization(wts=None,L1_decay=0.,L2_decay=0.):
		''' L1 or L2 regularization '''
		
		if wts is None:
			wts = self.wts_

		reg_loss = 0
		reg_loss += L1_decay*sum([T.sum(T.abs_(w)) for w in wts])
		reg_loss += L2_decay*sum([T.sum(T.abs_(w)) for w in wts])

		return reg_loss

def cross_entropy(y,y_prob):
	''' basic cross entropy loss function with optional regularization'''
	
	return T.mean(T.sum(-1.0*y*T.log(y_prob),axis=1))

def squared_error(y,y_prob):
	''' basic squared error loss function with optional regularization'''

	return T.mean(T.sum((y-y_prob)**2))

def max_norm_regularization(wts=None,c=3):
	
