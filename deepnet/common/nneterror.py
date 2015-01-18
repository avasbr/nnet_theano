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

def activ_err():
	err_msg = ('Invalid activation provided! Your choices are:'
				'\n(1) sigmoid'
				'\n(2) tanh'
				'\n(3) reLU'
				'\n(4) softmax')
	return err_msg

def config_req_err(item):
	return '%s is a required item, please check config file and include it'%item

def value_err(param,v):
	return '%s is not a valid value for %s; must be an integer'%(v,param)


