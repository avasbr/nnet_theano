# Currently very much a work in progress, but contains some pre-made
# spaces and objective functions - a lot of clean-up necessary, many
# things are
import numpy as np
import matplotlib.pyplot as plt
import theano
from hyperopt import hp, fmin, tpe, Trials, space_eval
from hyperopt.pyll.stochastic import sample
from math import log

from deepnet import MultilayerNet as mln
from deepnet import Autoencoder as ae
from deepnet.common import nnetact as na
from deepnet.common import nnetutils as nu
from deepnet.common import nnettrain as nt


class HyperparamOptimizer():

    def __init__(self, X, y, space_type='modern'):

        self.X = X
        self.y = y
        self.d = X.shape[1]
        self.k = y.shape[1]
        self.space_type = space_type

        # pre-training related fields
        self.pretrain_layer_1 = None
        self.pretrain_layer_2 = None
        self.curr_X = None
        self.pretrain_wts = []
        self.pretrain_bs = []

        # fine-tuning related fields

    def set_finetune_space(self, config_file):
        ''' Given the original deep net architecture, and a set of pretrained weights
        and biases, now fine tune it - we'll attempt to learn the right settings to 
        train this net as well'''

        # we know these fields won't change, so go ahead and set them as
        # defaults now
        model_params = nt.get_model_params(config_file)
        optim_params = nt.get_optim_params(config_file)
        default_model_params = {k: model_params[k] for k in ('num_hids', 'activs', 'd', 'k')}
        default_model_params.update({'loss_terms': ['cross_entropy']})
        default_optim_params = {k: model_params[k] for k in ('optim_method', 'optim_type')}

        # define the space of hyperparameters we wish to
        searchable_model_params = {'l1_reg': hp.choice('l1_reg', [None, hp.loguniform('l1_decay', log(1e-5), log(10))]),
                                   'l2_reg': hp.choice('l2_reg', [None, hp.loguniform('l2_decay', log(1e-5), log(10))])}
        searchable_optim_params = {'learn_rate': hp.uniform('learn_rate', 0, 1),
                                   'rho': hp.uniform('rho', 0, 1),
                                   'num_epochs': hp.qloguniform('num_epochs', log(10), log(5e3), 1),
                                   'batch_size': hp.quniform('batch_size', 128, 1024, 1),
                                   'init_method': hp.choice('init_method', ['gauss', 'fan-io']),
                                   'scale_factor': hp.uniform('scale_factor', 0, 1)}

        # combine the default and searchable parameters, and combine them into a dictionary to define the
        # full space - this is what will be passed into the objective function
        all_model_params = default_model_params.copy()
        all_model_params.update(searchable_model_params)
        all_optim_params = default_optim_params.copy()
        all_optim_params.update(searchable_optim_params)
        finetune_hyperspace = {
            'model_params': all_model_params, 'optim_params': all_optim_params}

        return finetune_hyperspace

    def compute_finetune_objective(self, hyperspace):
        ''' objective function for finetuning '''

        curr_model_params = {k: hyperspace['model_params'][k] for k in ('num_hids', 'activs', 'd',
                                                                        'k', 'loss_terms')}
        curr_optim_params = {k: optim_params['optim_params'][k] for k in ('learn_rate', 'rho', 'num_epochs',
                                                                          'batch_size', 'scale_factor')}

        # there's a little extra we need to do before this is completely ready
        if 'l1_decay' in hyperspace['model_params']:
            curr_model_params['loss_terms'].append('l1_reg')
            curr_model_params[
                'l1_decay':hyperspace['model_params']['l1_decay']]

        if 'l2_decay' in hyperspace['model_params']:
            curr_model_params['loss_terms'].append('l2_reg')
            curr_model_params[
                'l2_decay':hyperspace['model_params']['l2_decay']]

        if hyperspace['optim_params']['init_method'] == 0:
            curr_optim_params['init_method'] = 'gauss'
        else:
            curr_optim_params['init_method'] = 'fan-io'

        return self.compute_val_loss(curr_model_params, curr_optim_params,
                                     wts=self.pretrain_wts, bs=self.pretrain_bs)

    def learn_pretrain_settings(self, config_file):
        ''' automatically searches for the right settings to pre-train
        the model described in the configuration path '''

        # we will reuse this hyperspace (denoising autoencoder) for every pair
        # of input-output layers
        pretrain_hyperspace = {'pretrain_params': [
            {'corrupt_p': hp.uniform('corrupt_p', 0, 1)},
            {'l1_reg': hp.choice(
                'l1_reg', [None, hp.loguniform('l1_decay', log(1e-5), log(10))])},
            {'l2_reg': hp.choice(
                'l2_reg', [None, hp.loguniform('l2_decay', log(1e-5), log(10))])},
        ],
            'optim_params': [
            {'learn_rate': hp.uniform('learn_rate', 0, 1)},
            {'rho': hp.uniform('rho', 0, 1)},
            {'num_epochs': hp.qloguniform(
                'num_epochs', log(1e2), log(2000), 1)},
            {'batch_size': hp.quniform('batch_size', 128, 1024, 1)},
            {'init_method': hp.choice(
                'init_method', ['gauss', 'fan-io'])},
            {'scale_factor': hp.uniform(
                'scale_factor', 0, 1)}
        ]
        }

        # the final layer is trained normally, via simple softmax regression
        last_hyperspace = {'last_params': [
            {'l1_reg': hp.choice(
                'l1_reg', [None, hp.loguniform('l1_decay', log(1e-5), log(10))])},
            {'l2_reg': hp.choice(
                'l2_reg', [None, hp.loguniform('l2_decay', log(1e-5), log(10))])},
        ],
            'optim_params': [
            {'learn_rate': hp.uniform('learn_rate', 0, 1)},
            {'rho': hp.uniform('rho', 0, 1)},
            {'num_epochs': hp.qloguniform(
                'num_epochs', log(1e2), log(2000), 1)},
            {'batch_size': hp.quniform('batch_size', 128, 1024, 1)},
            {'init_method': hp.choice(
                'init_method', ['gauss', 'fan-io'])},
            {'scale_factor': hp.uniform(
                'scale_factor', 0, 1)}
        ]
        }

        # get the number of nodes per hidden layer so we know how to
        # initialize the autoencoders
        model_params = nt.get_model_params(config_file)
        pretrain_nodes = [self.d] + model_params['num_hids']

        best_pretrain_settings = []
        self.curr_X = self.X
        print 'Starting pre-training...'
        for l1, l2 in zip(pretrain_nodes[:-1], pretrain_nodes[1:]):
            self.pretrain_layer_1 = l1
            self.pretrain_layer_2 = l2
            print 'Layer 1: %i' % self.pretrain_layer_1
            print 'Layer 2: %i' % self.pretrain_layer_2
            best = fmin(self.compute_pretrain_objective, pretrain_hyperspace, algo=tpe.suggest,
                        max_evals=1)
            best_pretrain_settings.append(best)
            # this updates curr_X for the next pre-training layer
            self.pretrain_layer_with_settings(best)

        # the last layer is not pretrained with an autoencoder
        best = fmin(self.compute_last_objective, last_hyperspace, algo=tpe.suggest,
                    max_evals=1)
        best_pretrain_settings.append(best)

        return best_pretrain_settings

    def pretrain_layer_with_settings(self, best):
        ''' given a learned best-setting, train that layer, and then generate the next set of
        inputs for the next pre-training layer '''

        pretrain_params = {}
        optim_params = {}

        # TODO: there has to be a cleaner way to go from the output 'best' of hyperopt to
        # setting these parameters
        loss_terms = ['cross_entropy', 'corruption']
        if best['l1_reg'] != 0:
            loss_terms.append('l1_reg')
            pretrain_params['l1_decay'] = best['l1_decay']
        if best['l2_reg'] != 0:
            loss_terms.append('l2_reg')
            pretrain_params['l2_decay'] = best['l2_decay']

        # collect all hyperparameters for the model..
        pretrain_params['d'] = self.pretrain_layer_1
        pretrain_params['loss_terms'] = loss_terms
        pretrain_params['corrupt_type'] = 'mask'
        pretrain_params['corrupt_p'] = best['corrupt_p']
        pretrain_params['num_hids'] = [self.pretrain_layer_2]
        pretrain_params['activs'] = ['sigmoid', 'sigmoid']

        # ..and optimization
        optim_params['optim_method'] = 'RMSPROP'
        optim_params['optim_type'] = 'minibatch'
        optim_params['learn_rate'] = best['learn_rate']
        optim_params['rho'] = best['rho']
        optim_params['num_epochs'] = int(best['num_epochs'])
        optim_params['batch_size'] = int(best['batch_size'])
        if best['init_method'] == 0:
            optim_params['init_method'] = 'gauss'
        else:
            optim_params['init_method'] = 'fan-io'
        optim_params['scale_factor'] = best['scale_factor']

        print pretrain_params

        # pre-train the layer
        dae = ae.Autoencoder(**pretrain_params)
        dae.fit(self.curr_X, **optim_params)

        # we'll need these to initialize the final net for fine-tuning
        self.pretrain_wts.append(dae.wts_[0])
        self.pretrain_bs.append(dae.bs_[0])

        # and set the input to the next layer
        self.curr_X = dae.encode(self.curr_X)

    def compute_pretrain_objective(self, hyperspace):
        ''' objective function for pre-training layers '''

        # parse the hyperparams from the curr hyperspace
        curr_pretrain_params = {}
        curr_optim_params = {}

        # collect dictionaries into single dictionary
        for param in hyperspace['pretrain_params']:
            curr_pretrain_params.update(param)
        for param in hyperspace['optim_params']:
            if 'batch_size' in param:
                param['batch_size'] = int(param['batch_size'])
            if 'num_epochs' in param:
                param['num_epochs'] = int(param['num_epochs'])
            curr_optim_params.update(param)

        # collect number of hidden units and define activation functions
        num_hids = [self.pretrain_layer_2]
        activs = ['sigmoid', 'sigmoid']

        # set the loss terms
        loss_terms = ['cross_entropy', 'corruption']
        corrupt_p = curr_pretrain_params['corrupt_p']
        l1_decay = curr_pretrain_params['l1_reg']
        l2_decay = curr_pretrain_params['l2_reg']

        if l1_decay is not None:
            loss_terms.append('l1_reg')
        if l2_decay is not None:
            loss_terms.append('l2_reg')

        # multilayer parameters
        pretrain_params = {'d': self.pretrain_layer_1, 'num_hids': num_hids, 'activs': activs,
                           'loss_terms': loss_terms, 'corrupt_p': corrupt_p, 'corrupt_type': 'mask',
                           'l2_decay': l2_decay, 'l1_decay': l1_decay}

        # rmsprop parameters
        rmsprop_params = {'optim_method': 'RMSPROP', 'optim_type': 'minibatch'}
        rmsprop_params.update(curr_optim_params)

        print 'Pretraining parameters'
        print pretrain_params
        print 'Optimization parameters'
        print rmsprop_params

        return self.compute_val_reconstruction_loss(pretrain_params, rmsprop_params)
        # return self.compute_cv_loss(mln_params, rmsprop_params)

    def compute_last_objective(self, hyperspace):

        curr_last_params = {}
        curr_optim_params = {}

        for param in hyperspace['last_params']:
            curr_last_params.update(param)
        for param in hyperspace['optim_params']:
            if 'batch_size' in param:
                param['batch_size'] = int(param['batch_size'])
            if 'num_epochs' in param:
                param['num_epochs'] = int(param['num_epochs'])
            curr_optim_params.update(param)

        # collect number of hidden units and define activation functions
        num_hids = []  # the final layer is a
        activs = ['softmax']

        # set the loss terms
        loss_terms = ['cross_entropy']
        l1_decay = curr_last_params['l1_reg']
        l2_decay = curr_last_params['l2_reg']

        if l1_decay is not None:
            loss_terms.append('l1_reg')
        if l2_decay is not None:
            loss_terms.append('l2_reg')

        # multilayer parameters
        last_params = {'d': self.pretrain_layer_2, 'k': self.k, 'num_hids': num_hids, 'activs': activs,
                       'loss_terms': loss_terms, 'l2_decay': l2_decay, 'l1_decay': l1_decay}

        # rmsprop parameters
        rmsprop_params = {'optim_method': 'RMSPROP', 'optim_type': 'minibatch'}
        rmsprop_params.update(curr_optim_params)

        print 'Last layer parameters'
        print last_params
        print 'Optimization parameters'
        print rmsprop_params

        return self.compute_val_loss(last_params, rmsprop_params, X=self.curr_X, y=self.y)

    def set_multilayer_dropout_space(self):
        ''' defines a hyperspace for a "modern" neural networks: at least two layers with dropout + reLU '''

        # Force at least 2 layers, cuz we're modern
        min_layers = 2
        max_layers = 3

        # sets up the neural network
        nnets = [None] * (max_layers - min_layers + 1)

        for i, num_layers in enumerate(range(min_layers, max_layers + 1)):
            num_hids = [None] * num_layers
            for j in range(num_layers):
                num_hids[j] = hp.qloguniform(
                    'num_hid_%i%i' % (i, j), log(100), log(1000), 1)

            nnets[i] = num_hids

        # define the hyperparamater space to search
        hyperspace = {'mln_params': [
            {'arch': hp.choice('arch', nnets)},
            {'input_p': hp.uniform('ip', 0, 1)},
            {'hidden_p': hp.uniform('hp', 0, 1)},
            {'l1_reg': hp.choice(
                'l1_reg', [None, hp.loguniform('l1_decay', log(1e-5), log(10))])},
            {'l2_reg': hp.choice(
                'l2_reg', [None, hp.loguniform('l2_decay', log(1e-5), log(10))])},
        ],
            'optim_params': [
            {'learn_rate': hp.uniform('learn_rate', 0, 1)},
            {'rho': hp.uniform('rho', 0, 1)},
            {'num_epochs': hp.qloguniform(
                'num_epochs', log(1e2), log(2000), 1)},
            {'batch_size': hp.quniform('batch_size', 128, 1024, 1)},
            {'init_method': hp.choice(
                'init_method', ['gauss', 'fan-io'])},
            {'scale_factor': hp.uniform(
                'scale_factor', 0, 1)}
        ]
        }
        return hyperspace

    def compute_multilayer_dropout_objective(self, hyperspace):
        ''' parses the multilayer with dropout hyperspace and translates it into a loss value which
        we will use to search the space of hyperparams '''

        # parse the hyperparams from the curr hyperspace
        curr_mln_params = {}
        curr_optim_params = {}

        # collect dictionaries into single dictionary
        for param in hyperspace['mln_params']:
            curr_mln_params.update(param)
        for param in hyperspace['optim_params']:
            if 'batch_size' in param:
                param['batch_size'] = int(param['batch_size'])
            if 'num_epochs' in param:
                param['num_epochs'] = int(param['num_epochs'])
            curr_optim_params.update(param)

        # collect number of hidden units and define activation functions
        num_hids = list(curr_mln_params['arch'])
        activs = ['reLU'] * len(num_hids) + ['softmax']

        # set the loss terms
        loss_terms = ['cross_entropy', 'dropout']
        input_p = curr_mln_params['input_p']
        hidden_p = curr_mln_params['hidden_p']
        l1_decay = curr_mln_params['l1_reg']
        l2_decay = curr_mln_params['l2_reg']

        if l1_decay is not None:
            loss_terms.append('l1_reg')
        if l2_decay is not None:
            loss_terms.append('l2_reg')

        # multilayer parameters
        mln_params = {'d': self.d, 'k': self.k, 'num_hids': num_hids, 'activs': activs,
                      'loss_terms': loss_terms, 'l2_decay': l2_decay, 'l1_decay': l1_decay,
                      'input_p': input_p, 'hidden_p': hidden_p}

        # rmsprop parameters
        rmsprop_params = {'optim_method': 'RMSPROP', 'optim_type': 'minibatch'}
        rmsprop_params.update(curr_optim_params)

        print 'Multilayer net parameters'
        print mln_params
        print 'Optimization parameters'
        print rmsprop_params

        return self.compute_val_loss(mln_params, rmsprop_params)
        # return self.compute_cv_loss(mln_params, rmsprop_params)

    def set_old_space(self):
        ''' defines an old net from the 80s - simple sigmoid layers, nothing fancy'''

        min_layers = 1
        max_layers = 3

        # sets up the neural network
        nnets = [None] * (max_layers - min_layers + 1)

        for i, num_layers in enumerate(range(min_layers, max_layers + 1)):
            num_hids = [None] * num_layers
            for j in range(num_layers):
                num_hids[j] = hp.qloguniform(
                    'num_hid_%i%i' % (i, j), log(10), log(100), 1)

            nnets[i] = num_hids

        # define the hyperparamater space to search
        hyperspace = {'mln_params': [
            {'arch': hp.choice('arch', nnets)},
            {'l1_reg': hp.choice(
                'l1_reg', [None, hp.loguniform('l1_decay', log(1e-5), log(10))])},
            {'l2_reg': hp.choice(
                'l2_reg', [None, hp.loguniform('l2_decay', log(1e-5), log(10))])},
        ],
            'optim_params': [
            {'learn_rate': hp.uniform('learn_rate', 0, 1)},
            {'rho': hp.uniform('rho', 0, 1)},
            {'num_epochs': hp.qloguniform(
                'num_epochs', log(10), log(5e3), 1)},
            {'batch_size': hp.quniform('batch_size', 128, 1024, 1)},
            {'init_method': hp.choice(
                'init_method', ['gauss', 'fan-io'])},
            {'scale_factor': hp.uniform(
                'scale_factor', 0, 1)}
        ]
        }
        return hyperspace

    def compute_old_objective(self, hyperspace):
        ''' objective function that takes in a hyperspace and returns a cost/value '''

        # parse the hyperparams from the curr hyperspace
        curr_mln_params = {}
        curr_optim_params = {}

        # collect dictionaries into single dictionary
        for param in hyperspace['mln_params']:
            curr_mln_params.update(param)
        for param in hyperspace['optim_params']:
            if 'batch_size' in param:
                param['batch_size'] = int(param['batch_size'])
            if 'num_epochs' in param:
                param['num_epochs'] = int(param['num_epochs'])
            curr_optim_params.update(param)

        # collect number of hidden units and activation functions
        num_hids = list(curr_mln_params['arch'])
        activs = ['sigmoid'] * len(num_hids) + ['softmax']

        # set the loss terms
        loss_terms = ['cross_entropy']
        l1_decay = curr_mln_params['l1_reg']
        l2_decay = curr_mln_params['l2_reg']

        if not l1_decay is None:
            loss_terms.append('l1_reg')
        if not l2_decay is None:
            loss_terms.append('l2_reg')

        # multilayer parameters
        mln_params = {'d': self.d, 'k': self.k, 'num_hids': num_hids, 'activs': activs,
                      'loss_terms': loss_terms, 'l2_decay': l2_decay, 'l1_decay': l1_decay}
        # rmsprop parameters
        rmsprop_params = {'optim_method': 'RMSPROP', 'optim_type': 'minibatch'}
        rmsprop_params.update(curr_optim_params)

        print 'Multilayer net parameters'
        print mln_params
        print 'Optimization parameters'
        print rmsprop_params

        return self.compute_val_loss(mln_params, rmsprop_params)
        # return self.compute_cv_loss(mln_params, rmsprop_params)

    #-------------------Functions for computing validation---------------

    def compute_val_reconstruction_loss(self, pretrain_params, optim_params, p=0.8):
        ''' Reconstruction loss '''

        X_tr, X_val = nu.split_train_val(self.curr_X, p)
        nnet = ae.Autoencoder(**pretrain_params)
        nnet.fit(X_tr, **optim_params)

        re_val_loss = float(nnet.compute_reconstruction_loss(X_val))
        print 'Reconstruction loss on Validation set:', re_val_loss

        return re_val_loss

    def compute_val_loss(self, mln_params, optim_params, X=None, y=None, p=0.8, wts=None, bs=None):
        ''' Uses a single train/val split to compute the loss '''

        if X is None:
            X = self.X
        if y is None:
            y = self.y

        X_tr, y_tr, X_val, y_val = nu.split_train_val(X, p, y=y)
        nnet = mln.MultilayerNet(**mln_params)

        # adding optional weights/biases here allows for fine-tuning, if needed
        nnet.fit(X_tr, y_tr, wts=wts, bs=bs, **optim_params)
        val_loss = float(nnet.compute_test_loss(X_val, y_val))

        print 'Validation loss:', val_loss

        return val_loss

    def compute_cv_loss(self, mln_params, optim_params, k_cv=5):
        ''' Uses k-fold cross-val to compute the average loss '''

        # get the indices of the splits
        cv_splits = nu.split_k_fold_cross_val(self.X, k_cv=k_cv, y=self.y)

        val_loss = 0.  # needed to accumulate the validation loss

        for i, split in enumerate(cv_splits):
            print 'Cross-validation iteration:', i + 1
            # get the training and validation for this split
            X_tr, y_tr, X_val, y_val = split
            # initialize the neural network
            nnet = mln.MultilayerNet(**mln_params)
            nnet.fit(X_tr, y_tr, **optim_params)  # fit to the training
            # add to the validation loss
            val_loss += float(nnet.compute_test_loss(X_val, y_val))

        avg_loss = 1. * val_loss / k_cv  # compute the average
        print 'Average loss:', avg_loss

        return avg_loss

    def run_hyperopt(self, config_file=None):
        ''' convenience function for running hyperopt '''

        best = None
        if self.space_type == 'modern':
            hyperspace = self.set_multilayer_dropout_space()
            best = fmin(self.compute_multilayer_dropout_objective, hyperspace, algo=tpe.suggest,
                        max_evals=100)
            return best
        elif self.space_type == 'old':
            hyperspace = self.set_old_space()
            best = fmin(self.compute_old_objective, hyperspace, algo=tpe.suggest,
                        max_evals=100)
            return best
        elif self.space_type == 'pretrain':
            if config_file is None:
                sys.exit(
                    'Cannot pre-train a network without its original config file')
            else:
                best = self.learn_pretrain_settings(config_file)
        else:
            sys.exit(
                'Space type not specified correctly, your choices are: "modern","old", or "pretrain"')

        return best
