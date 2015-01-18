import numpy as np
import theano
import theano.tensor as T
from deepnet.common import nnetutils as nu
import copy

def maxnorm(w,c):
	''' clamping function which restricts the weight vector to lie on L2 ball of radius c '''
	
	l2n = T.sum(w**2,axis=0)
	return w*(c/T.sqrt(l2n)*(l2n > c**2) + 1.*(l2n < c**2))

def sgd(params,grad_params,learn_rate=0.1,max_norm=False,c=5):
	''' Assuming all the data can fit in memory, runs stochastic gradient descent with optional max-norm
	regularization. This tends to work well with dropout + rectified linear activation functions '''
	
	updates = []
	for param,grad_param in zip(params,grad_params):
		param_ = param - learn_rate*grad_param
		
		# there's probably a better way to check if this is a weight matrix...
		if max_norm and param_.get_value().ndim == 2:  
			param_ = maxnorm(param_,c)

		updates.append((param,param_))

	return updates

def rmsprop(params,grad_params,learn_rate=0.001,rho=0.9,eps=1e-6,max_norm=False,c=3):
	''' Geoff hinton's '"RMSprop" algorithm - RPROP for mini-batches
	
	Parameters:
	----------

	param: params - model parameters
	type: list of theano shared variables

	param: grad_params - derivative of the loss with respect to the model parameters
	type: list of theano variables

	param: learn_rate - learning rate for rmsprop
	type: float

	param: rho - momentum term

	param: eps - fudge factor [need ref here]
	type: float

	param: max_norm - flag to incidate that weights will be L2-norm constrained
	type: boolean

	param: c - L2-norm constraint for max_norm regularization
	type: float

	'''
	updates = []
	
	for param,grad_param in zip(params,grad_params):

		# accumulated gradient
		acc_grad_param = theano.shared(nu.floatX(np.zeros(param.get_value().shape))) # initial value
		acc_grad_param_ = rho*acc_grad_param + (1-rho)*grad_param**2
		
		# parameter update
		param_ = param - learn_rate*grad_param/T.sqrt(acc_grad_param_ + eps)
		
		# there's probably a better way to check if this is a weight matrix...
		if max_norm and param.get_value().ndim == 2:
			param_ = maxnorm(param_,c)

		# collected updates of both the parameter and the accumulated gradient
		updates.append((acc_grad_param,acc_grad_param_))
		updates.append((param,param_))

	return updates

def adagrad(params,grad_params,learn_rate=1.,eps=1e-6,max_norm=False,c=5):
	''' adaptive gradient method - typically works better than vanilla SGD and has some 
	nice theoretical guarantees

	Parameters:
	----------

	param: params - model parameters
	type: list of theano shared variables

	param: grad_params - derivative of the loss with respect to the model parameters
	type: list of theano variables

	param: learn_rate - 'master' learning rate for the adagrad algorithm
	type: float

	param: eps - fudge factor [need ref here]
	type: float

	param: max_norm - flag to incidate that weights will be L2-norm constrained
	type: boolean

	param: c - L2-norm constraint for max_norm regularization
	type: float

	Returns:
	--------
	None

	Updates:
	--------
	wts,bs
	'''
		
	updates = []
	for param,grad_param in zip(params,grad_params):
		
		# accumulated gradient
		acc_grad_param = theano.shared(nu.floatX(np.zeros(param.get_value().shape)))
		acc_grad_param_ = acc_grad_param + grad_param**2
		
		# parameter update
		param_ = param - learn_rate*grad_param/T.sqrt(acc_grad_param_ + eps)

		# there's probably a better way to check if this is a weight matrix...
		if max_norm and param.get_value().ndim == 2:
			param_ = maxnorm(param_,c)
		
		# collected updates of both the parameter and the accumulated gradient
		updates.append((acc_grad_param,acc_grad_param_))
		updates.append((param,param_))

	return updates