
def model_error():
	err_msg = ('Not a valid neural network type! Your choices are:'
		'\n(1) MultilayerNet'
		'\n(2) Autoencoder'
		'\n(3) StackedAutoencoder')

	return err_msg

def method_err():
	err_msg = ('No (valid) method provided to fit! Your choices are:'
				'\n(1) SGD: vanilla stochastic gradient descent'+
				'\n(2) ADAGRAD: ADAptive GRADient learning'+
				'\n(3) RMSPROP: Hintons mini-batch version of RPROP'+
				'\n(4) L-BFGS-B: limited-memory lbfgs'+
				'\n(5) CG: conjugate gradient')
	return err_msg

def opt_type_err():
	err_msg = ('No valid optimization type specified! Your choices are:'
				'\n(1) minibatch'+
				'\n(2) fullbatch')
	return err_msg

def opt_type_opt_method_mismatch(opt_type,opt_method):
	return '%s cannot be applied to %s learning'%(opt_method,opt_type)

def config_err():
	pass
