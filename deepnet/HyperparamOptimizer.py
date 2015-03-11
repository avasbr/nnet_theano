# Currently very much a work in progress, but contains some pre-made
# spaces for convenience
import numpy as np
import matplotlib.pyplot as plt
import theano
from hyperopt import hp, fmin, tpe, Trials, space_eval
from hyperopt.pyll.stochastic import sample
from math import log

from deepnet import MultilayerNet as mln
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
        # since these correspond to set values that are not sampled,
        # we'll just keep track of them as we loop through the layers.
        # this is specific only to pretraining
        self.pretrain_layer_1 = None
        self.pretrain_layer_2 = None

    def learn_pretrain_settings(self, config_path):
        ''' automatically searches for the right settings to pre-train
        the model described in the configuration path '''

        # we will reuse this hyperspace (denoising autoencoder) for every pair of input-output layers
        pretrain_hyperspace = ({'pretrainer_params': [
            {'corrput_p': hp.uniform('cp', 0, 1)},
            {'l1_reg': hp.choice(
                'l1_lambda', [None, hp.loguniform('l1_decay', log(1e-5), log(10))])},
            {'l2_reg': hp.choice(
                'l2_lambda', [None, hp.loguniform('l2_decay', log(1e-5), log(10))])},
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
        })

        last_hyperspace = ()

        # get the number of nodes per hidden layer so we know how to
        # initialize the autoencoders
        model_params = nt.get_model_params(config_path)
        pretrain_layers = [self.d] + model_params['num_hids']

        best_pretrain_settings = []
        
        for l1, l2 in zip(pretrain_layers[:-1], pretrain_layers[1:]):
            self.pretrain_layer_1 = l1
            self.pretrain_layer_2 = l2
            best = fmin(self.compute_pretrain_layer_objective, pretrain_hyperspace, algo=tpe.suggest,
                        max_evals=100)
            best_pretrain_settings.append(best)

        # the last layer is not pretrained with an autoencoder
        best = fmin(self.compute_last_layer_objective, last_hyperspace, algo=tpe.suggest,
            max_evals=100)
        best_pretrain_settings.append(best)

        return best_pretrain_settings

    #TODO: WRITE THESE
    def compute_last_layer_objective(self, hyperspace):
        pass
    def compute_pretrain_layer_objective(self, hyperspace):
        pass

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
        hyperspace = ({'mln_params': [
            {'arch': hp.choice('arch', nnets)},
            {'input_p': hp.uniform('ip', 0, 1)},
            {'hidden_p': hp.uniform('hp', 0, 1)},
            {'l1_reg': hp.choice(
                'l1_lambda', [None, hp.loguniform('l1_decay', log(1e-5), log(10))])},
            {'l2_reg': hp.choice(
                'l2_lambda', [None, hp.loguniform('l2_decay', log(1e-5), log(10))])},
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
        })
        return hyperspace

    def compute_multilayer_dropout_objective(self, hyperspace):
        ''' parses the multilayer with dropout hyperspace and translates it into a loss value which
        we will use to search the space of hyperparams '''

        # parse the hyperparams from the sampled hyperspace
        sampled_mln_params = {}
        sampled_optim_params = {}

        # collect dictionaries into single dictionary
        for param in hyperspace['mln_params']:
            sampled_mln_params.update(param)
        for param in hyperspace['optim_params']:
            if 'batch_size' in param:
                param['batch_size'] = int(param['batch_size'])
            if 'num_epochs' in param:
                param['num_epochs'] = int(param['num_epochs'])
            sampled_optim_params.update(param)

        # collect number of hidden units and define activation functions
        num_hids = list(sampled_mln_params['arch'])
        activs = ['reLU'] * len(num_hids) + ['softmax']

        # set the loss terms
        loss_terms = ['cross_entropy', 'dropout']
        input_p = sampled_mln_params['input_p']
        hidden_p = sampled_mln_params['hidden_p']
        l1_decay = sampled_mln_params['l1_reg']
        l2_decay = sampled_mln_params['l2_reg']

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
        rmsprop_params.update(sampled_optim_params)

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
                'l1_lambda', [None, hp.loguniform('l1_decay', log(1e-5), log(10))])},
            {'l2_reg': hp.choice(
                'l2_lambda', [None, hp.loguniform('l2_decay', log(1e-5), log(10))])},
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

        # parse the hyperparams from the sampled hyperspace
        sampled_mln_params = {}
        sampled_optim_params = {}

        # collect dictionaries into single dictionary
        for param in hyperspace['mln_params']:
            sampled_mln_params.update(param)
        for param in hyperspace['optim_params']:
            if 'batch_size' in param:
                param['batch_size'] = int(param['batch_size'])
            if 'num_epochs' in param:
                param['num_epochs'] = int(param['num_epochs'])
            sampled_optim_params.update(param)

        # collect number of hidden units and activation functions
        num_hids = list(sampled_mln_params['arch'])
        activs = ['sigmoid'] * len(num_hids) + ['softmax']

        # set the loss terms
        loss_terms = ['cross_entropy']
        l1_decay = sampled_mln_params['l1_reg']
        l2_decay = sampled_mln_params['l2_reg']

        if not l1_decay is None:
            loss_terms.append('l1_reg')
        if not l2_decay is None:
            loss_terms.append('l2_reg')

        # multilayer parameters
        mln_params = {'d': self.d, 'k': self.k, 'num_hids': num_hids, 'activs': activs,
                      'loss_terms': loss_terms, 'l2_decay': l2_decay, 'l1_decay': l1_decay}
        # rmsprop parameters
        rmsprop_params = {'optim_method': 'RMSPROP', 'optim_type': 'minibatch'}
        rmsprop_params.update(sampled_optim_params)

        print 'Multilayer net parameters'
        print mln_params
        print 'Optimization parameters'
        print rmsprop_params

        return self.compute_val_loss(mln_params, rmsprop_params)
        # return self.compute_cv_loss(mln_params, rmsprop_params)

    def compute_val_loss(self, mln_params, optim_params, p=0.8):
        ''' Uses a single train/val split to compute the loss '''

        X_tr, y_tr, X_val, y_val = nu.split_train_val(self.X, p, y=self.y)
        nnet = mln.MultilayerNet(**mln_params)
        nnet.fit(X_tr, y_tr, **optim_params)

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

    def run_hyperopt(self):
        ''' convenience function for running hyperopt '''

        best = None

        if self.space_type == 'modern':
            hyperspace = self.set_multilayer_dropout_space()
            best = fmin(self.compute_multilayer_dropout_objective, hyperspace, algo=tpe.suggest,
                        max_evals=100)
        elif self.space_type == 'old':
            hyperspace = self.set_old_space()
            best = fmin(self.compute_old_objective, hyperspace, algo=tpe.suggest,
                        max_evals=100)
        else:
            sys.exit(
                'Space type not specified correctly, your choices are: "modern" or "old"')

        return best
