# TODO: should turn all this into a class, call it "ConfigTrainer" or something
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

    for key, value in model_params.iteritems():
        if not (key == 'corrupt_type'):
            model_params[key] = ast.literal_eval(value)
    return model_params


def clean_optim_params(optim_params):
    ''' parses the optimization parameters '''

    for key, value in optim_params.iteritems():
        if not (key == 'init_method' or key == 'optim_method' or key == 'optim_type'):
            optim_params[key] = ast.literal_eval(value)
    return optim_params


def get_model_params(config_file):
    ''' returns just the model parameters given a config file. turns out I need this
    for a hyperparameter optimizer '''
    cfg_parser = ConfigParser()
    cfg_parser.read(config_file)
    return clean_model_params(dict(cfg_parser.items('model_params')))

def get_optim_params(config_file):
    ''' returns just the optimization parameters given a config file '''
    cfg_parser = ConfigParser():
    cfg_parser.read(config_file)
    return clean_model_params(dict(cfg_parser.items('optim_params')))


def train_single_net(model_type, model_params, optim_params, X_tr, y_tr=None, X_val=None, y_val=None, wts=None, bs=None):
    ''' defines a single neural network given a model type and parameters '''

    # get the model and optimization parameters in a format for ingestion
    # TODO: better to do this in the train_nnet function
    model_params = clean_model_params(dict(model_params))
    optim_params = clean_optim_params(dict(optim_params))

    nnet = None

    # construct the model based on the specified architecture
    if model_type == 'MultilayerNet':
        nnet = mln.MultilayerNet(**model_params)
        nnet.fit(X_tr, y_tr, X_val=X_val, y_val=y_val,
                 wts=wts, bs=bs, **optim_params)
    elif model_type == 'Autoencoder':
        nnet = ae.Autoencoder(**model_params)
        nnet.fit(X_tr, wts=wts, bs=bs, **optim_params)
    else:
        sys.exit(ne.model_error())

    # train the neural network
    return nnet


def train_nnet(config_file, X_tr, y_tr=None, X_val=None, y_val=None, wts=None, bs=None):
    ''' parses a config file to initialize a neural network '''

    # define the parser
    cfg_parser = ConfigParser()
    cfg_parser.read(config_file)

    # get the model type, model parameters and optimization parameters
    model_type = cfg_parser.get('model_type', 'arch')

    if model_type == 'Pretrainer':

        X_in = X_tr
        num_trainers = (len(cfg_parser.sections()) - 1) / 3
        nnet_wts = [None] * num_trainers
        nnet_bs = [None] * num_trainers
        nnet = None

        for idx in range(1, num_trainers + 1):

            # parse out the next pre-trainer...
            curr_model_type = cfg_parser.get('model_type_' + str(idx), 'arch')
            curr_model_params = cfg_parser.items('model_params_' + str(idx))
            curr_optim_params = cfg_parser.items('optim_params_' + str(idx))

            # train it...
            if curr_model_type == 'Autoencoder':
                nnet = train_single_net(
                    curr_model_type, curr_model_params, curr_optim_params, X_in)
                X_in = nnet.encode(X_in)

            # this can really only happen once, so if y_tr is specified, it's
            # probably for the last layer
            elif curr_model_type == 'MultilayerNet':
                nnet = train_single_net(
                    curr_model_type, curr_model_params, curr_optim_params, X_in, y_tr=y_tr)

            # store the pre-trained weights and biases
            pt_wts, pt_bs = nnet.get_weights_and_biases()
            nnet_wts[idx - 1] = pt_wts
            nnet_bs[idx - 1] = pt_bs

        return nnet_wts, nnet_bs

    else:
        model_params = cfg_parser.items('model_params')
        optim_params = cfg_parser.items('optim_params')
        return train_single_net(model_type, model_params, optim_params, X_tr, y_tr=y_tr, wts=wts, bs=bs)

# TODO: THROW ALL THIS ERROR CHECKING IN THE NEURAL NETWORK CORE CONSTRUCTOR

    # define the required fields and check against them - these MUST be present
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
    # L-BFGS-B and CG, in these implementations, cannot be used with mini-batches,
    # although in theory they can
    # 		sys.exit(ne.opt_type_opt_method_mismatch(optim_params['opt_type'],optim_params['method']))
    # 	else:
    # 		nnet.fit(X_tr,y_tr,**optim_params)
    # elif optim_params['opt_type'] == 'fullbatch':
    # 	nnet.fit(X_tr,y_tr,**optim_params)
    # else:
    # 	sys.exit(ne.opt_type_err())
