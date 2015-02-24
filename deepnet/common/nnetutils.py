import theano
import theano.tensor as T
import numpy as np

def split_train_val(X,p,y=None):
	''' splits into training and validation sets '''
	
	m = X.shape[0] # number of examples in X
	idx = np.random.permutation(m) # shuffles the indices
	m_tr = m*p # number of examples for the training set
	
	# split the data into training and validation sets
	tr_idx = idx[:m_tr]
	val_idx = idx[m_tr:]
	X_tr = X[tr_idx]
	X_val = X[val_idx]

	# if y has also been provided, split those accordingly
	if y is not None:
		y_tr = y[tr_idx]
		y_val = y[val_idx]
		return X_tr,y_tr,X_val,y_val
		
	return X_tr,X_val

# t_* are for unrolling theano variables
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
	v = np.array([],dtype=theano.config.floatX)
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

def floatX(X):
	return np.asarray(X,dtype=theano.config.floatX)

def pretty_print(header,params):
	print header
	print '-'*len(header)
	for k,v in params.iteritems():
		print k,':',v
	print '\n'

def split_k_fold_cross_val(X,k_cv=5,y=None):
	''' Returns a list of tuples consisting of cross-val indices '''
	
	# set up the indices
	m = X.shape[0]
	batch_size = int(m/k_cv) # round down
	leftover = m - batch_size*k_cv
	batch_idx = range(0,m+1,batch_size)
	batch_idx[-1] += leftover

	# split the data 
	idx = list(np.random.permutation(m))
	cross_val_splits = [None]*k_cv
	for i,(start,end) in enumerate(zip(batch_idx[:-1],batch_idx[1:])):
		te_idx = idx[start:end] # held-out slice
		tr_idx = idx[0:start]+idx[end:] # training slice
		
		# set the training/held-out data for this data 
		X_tr = X[tr_idx]
		X_te = X[te_idx]
		if y is not None:
			y_tr = y[tr_idx]
			y_te = y[te_idx] 
			cross_val_splits[i] = (X_tr,y_tr,X_te,y_te)
		else:
			cross_val_splits[i] = (X_tr,X_te)

	return cross_val_splits



	
