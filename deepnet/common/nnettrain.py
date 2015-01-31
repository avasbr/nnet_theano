from deepnet import NeuralNetworkCore
from deepnet import MultilayerNet as mln
from deepnet import Autoencoder as ae
from deepnet.common import nnetact as na
from deepnet.common import nneterror as ne
from deepnet.common import nnetutils as nu
import sys
import ast
from ConfigParser import SafeConfigParser, ConfigParser

def clean_model_params(model_params):
	''' parses the model parameters '''
	
	for key,value in model_params.iteritems():
		if not (key == 'corrupt_type'):
			model_params[key] = ast.literal_eval(value)

	return model_params

def clean_optim_params(optim_params):
	''' parses the optimization parameters '''
	 for key,value in optim_params.iteritems():
		if not (key == 'init_method' or key == 'optim_method' or key == 'optim_type'):
			optim_params[key] = ast.literal_eval(value)

	return optim_params

def train_single_net(model,model_params,optim_params):
	

def train_nnet(config_file,X_tr,y_tr=None,X_val=None,y_val=None):
	''' parses a config file to initialize a neural network '''

	# define the parser
	cfg_parser = ConfigParser()
	cfg_parser.read(config_file)

	# get the model type, model parameters and optimization parameters
	model = cfg_parser.get('model_type','arch')
	if arch == 'Pretrainer':
		num_trainers = (len(cfg_parser.get_sections())-1)/3
		for idx in num_trainers:
			curr_model_type = 'model_type_'+idx
			curr_model_param = 'model_params_'+idx
			curr_optim_param = 'optim_params_'+idx

	else:
		model_params = clean_model_params(dict(cfg_parser.items('model_params')))
		optim_params = clean_model_params(dict(cfg_parser.items('optim_params')))

		# start constructing the neural network model
		nnet = None

		# construct the model based on the specified architecture
		if model == 'MultilayerNet':
			nnet = mln.MultilayerNet(**model_params)
			nnet.fit(X_tr,y_tr,X_val=X_val,y_val=y_val,**optim_params)
		elif model == 'Autoencoder':
			nnet = ae.Autoencoder(**model_params)
			nnet.fit(X_tr,**optim_params)
		else:
			sys.exit(ne.model_error())

		# train the neural network
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
