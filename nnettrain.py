import NeuralNetworkCore
import MultilayerNet as mln
import Autoencoder as ae
import nnetact as na
import nneterror as ne
import sys
import ast
from ConfigParser import SafeConfigParser, ConfigParser

def train_nnet(X_tr,y_tr,config_file,X_val=None,y_val=None):
	''' parses a config file to initialize a neural network '''

	# define the parser
	cfg_parser = ConfigParser()
	cfg_parser.read(config_file)

	# get the model type, model parameters and optimization parameters
	model = cfg_parser.get('model_type','arch')
	model_params = dict(cfg_parser.items('model_params'))
	optim_params = dict(cfg_parser.items('optim_params'))

	# start constructing the neural network model
	nnet = None

	# infer the types of all the parameters
	for key,value in model_params.iteritems():
		model_params[key] = ast.literal_eval(value)
	
	# construct the model based on the specified architecture
	if model == 'MultilayerNet':
		nnet = mln.MultilayerNet(**model_params)
	elif model == 'Autoencoder':
		pass
	elif model == 'StackedDenoisingAutoencoder':
		pass
	else:
		sys.exit(ne.model_error())

	# now parse the optimization methods
	for key,value in optim_params.iteritems():
		if not key == 'method' and not key == 'opt_type':
			optim_params[key] = ast.literal_eval(value)

	# train the neural network
	nnet.fit(X_tr,y_tr,X_val=X_val,y_val=y_val,**optim_params)

	return nnet

#TODO: THROW ALL THIS ERROR CHECKING IN THE NEURAL NETWORK CORE CONSTRUCTOR

	# # define the required fields and check against them - these MUST be present
	# req_model_set = set(['mod_type','d','k','num_hids','activs','loss_terms'])
	# req_optim_set = set(['method','opt_type','num_epochs'])
	
	# for item in req_model_set:
	# 	if item not in model_params:
	# 		sys.exit(ne.config_req_err(item))
	# for item in req_optim_set:
	# 	if item not in optim_params:
	# 		sys.exit(ne.config_req_err(item))


	# if optim_params['opt_type'] == 'minibatch':
	# 	if optim_params['method'] == 'L-BFGS-B' or optim_params['method'] == 'CG':
	# 		# L-BFGS-B and CG, in these implementations, cannot be used with mini-batches,
	# 		# although in theory they can
	# 		sys.exit(ne.opt_type_opt_method_mismatch(optim_params['opt_type'],optim_params['method']))
	# 	else:
	# 		nnet.fit(X_tr,y_tr,**optim_params)
	# elif optim_params['opt_type'] == 'fullbatch':
	# 	nnet.fit(X_tr,y_tr,**optim_params)
	# else:
	# 	sys.exit(ne.opt_type_err())