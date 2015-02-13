import theano
import theano.tensor as T
import numpy as np

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

def split_k_fold_cross_val(X,y,k=10):
	''' Returns a list of tuples consisting of cross-val indices '''
	# set up the indices
	m = X.shape[0]
	batch_size = int(m/k) # round down
	leftover = m - batch_size*k
	batch_idx = range(0,m+1,m/k)
	batch_idx[-1] += leftover

	# split the data 
	idx = list(np.random.permutation(m))
	cross_val_splits = [None]*k
	for i,(start,end) in enumerate(zip(batch_idx[:-1],batch_idx[1:])):
		te_idx = idx[start:end] # held-out slice
		tr_idx = idx[0:start]+idx[end:] # training slice
		X_tr = X[tr_idx] 
		y_tr = y[tr_idx]
		X_te = X[te_idx]
		y_te = y[te_idx] 
		cross_val_splits[i] = (X_tr,y_tr,X_te,y_te)

	return cross_val_splits



	
