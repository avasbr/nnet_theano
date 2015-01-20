import theano
import theano.tensor as T

def regularization(wts,L1_decay=None,L2_decay=None):
	''' L1 and/or L2 regularization 

	Parameters:
	-----------
	param: wts - weights
	type: theano shared matrix

	param: L1_decay - L1 decay term
	type: float

	param: L2_decay - L2 decay term
	type: float

	Returns:
	--------
	param: reg_loss - regularization loss term
	type: theano scalar

	'''

	reg_loss = 0
	if L1_decay is not None:
		reg_loss += L1_decay*sum([T.sum(T.abs_(w)) for w in wts])
	if L2_decay is not None:
		reg_loss += L2_decay*sum([T.sum(T.abs_(w)) for w in wts])

	return reg_loss

def cross_entropy(y,y_pred):
	''' Basic cross entropy loss function

	Parameters:
	-----------
	param: y - true labels
	type: theano matrix

	param: y_pred - predicted labels
	type: theano matrix

	Returns:
	--------
	param: [expr] - cross entropy loss
	type: theano scalar

	'''
	# not sure if this is more numerically stable or what...
	return T.mean(T.nnet.categorical_crossentropy(y_pred,y))
	#return T.mean(T.sum(-1.0*y*T.log(y_pred),axis=1))

def squared_error(y,y_pred):
	''' basic squared error loss function

	Parameters:
	-----------
	param: y - true labels
	type: theano matrix

	param: y_pred - predicted labels
	type: theano matrix

	Returns:
	--------
	param: [expr] - squared loss
	type: theano scalar
	'''

	return T.mean(T.sum((y-y_pred)**2))

def sparsity(act,beta=None,rho=None):
	''' Sparsity term used to enforce sparsity in the activations of hidden units for
		autoencoders 

	Parameters:
	-----------
	param: act - activation values
	type: theano matrix

	param: beta - sparsity coefficient
	type: float

	param: rho - sparsity level
	type: float

	'''
		
	sparse_loss = 0

	if beta is not None and rho is not None:
		avg_act = T.mean(act,axis=0)
		sparse_loss = beta*T.sum(rho*T.log(rho/avg_act)+(1-rho)*T.log((1-rho)/(1-avg_act)))
	
	return sparse_loss
