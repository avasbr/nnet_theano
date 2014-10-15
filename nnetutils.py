import theano.Tensor as T

def sigmoid(z):
	'''Computes the element-wise logit of z'''
	return 1./(1. + T.exp(-1*z))
	

	