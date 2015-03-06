# Currently very much a work in progress
import numpy as np
import matplotlib.pyplot as plt
import theano
from hyperopt import hp, fmin, tpe, Trials, space_eval
from hyperopt.pyll.stochastic import sample
from math import log

from deepnet import MultilayerNet as mln
from deepnet.common import nnetact as na
from deepnet.common import nnetutils as nu


class HyperparamOptimizer():

    def __init__(self, X, y):

        self.X = X
        self.y = y
        self.d = X.shape[1]
        self.k = y.shape[1]

    def get_hyperspace_2(self):
        ''' defines the hyperspace and return it; all modifications should go here. there really
        isn't a need to make it a function '''

        # Multilayer nnet spaces
        max_layers = 4

        # sets up the neural network
        for num_layers in range(1, max_layers):
            activs = [None] * num_layers
            num_hids = [None] * num_layers

            # set the activation function choice per layer
            for i in range(num_layers):
                activs[i] = hp.choice('activ_%i' % i, ['sigmoid', 'reLU'])
                num_hids[i] = hp.qloguniform(
                    'num_hid_%i' % i, log(10), log(5000), 1)

        # define the hyperparamater space to search
        hyperspace = {'mln_params': [
            {'num_hids': num_hids},
            {'activs': activs},
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
                'num_epochs', log(1e2), log(5000), 1)},
            {'batch_size': hp.quniform('batch_size', 128, 1024, 1)},
            {'init_method': hp.choice(
                'init_method', ['gauss', 'fan-io'])},
            {'scale_factor': hp.uniform(
                'scale_factor', 0, 1)}
        ]
        }
        return hyperspace

    def get_hyperspace(self):
        ''' defines the hyperspace and return it; all modifications should go here. there really
        isn't a need to make it a function '''

        # Multilayer nnet spaces
        max_layers = 4

        # sets up the neural network
        for num_layers in range(1, max_layers):
            activs = [None] * num_layers
            num_hids = [None] * num_layers

            # set the activation function choice per layer
            for i in range(num_layers):
                activs[i] = hp.choice('activ_%i' % i, ['sigmoid', 'reLU'])
                num_hids[i] = hp.qloguniform(
                    'num_hid_%i' % i, log(10), log(5000), 1)

        # define the hyperparamater space to search
        hyperspace = {'mln_params': [
            {'num_hids': num_hids},
            {'activs': activs},
            {'dropout': hp.choice('dropout', [
                None,
                {'input_p': hp.uniform(
                    'ip', 0, 1), 'hidden_p': hp.uniform('hp', 0, 1)}
            ])
            },
            {'l1_reg': hp.choice(
                'l1_lambda', [None, hp.loguniform('l1_decay', log(1e-5), log(10))])},
            {'l2_reg': hp.choice(
                'l2_lambda', [None, hp.loguniform('l2_decay', log(1e-5), log(10))])},
        ],
            'optim_params': [
            {'learn_rate': hp.uniform('learn_rate', 0, 1)},
            {'rho': hp.uniform('rho', 0, 1)},
            {'num_epochs': hp.qloguniform(
                'num_epochs', log(10), log(1e4), 1)},
            {'batch_size': hp.quniform('batch_size', 128, 1024, 1)},
            {'init_method': hp.choice(
                'init_method', ['gauss', 'fan-io'])},
            {'scale_factor': hp.uniform(
                'scale_factor', 0, 1)}
        ]
        }
        return hyperspace

    def compute_cv_loss(self, mln_params, optim_params, k_cv=5):
        ''' Uses k-fold cross-val to compute the average loss '''

        # get the indices of the splits
        cv_splits = nu.split_k_fold_cross_val(self.X, k_cv=k_cv, y=self.y)

        val_loss = 0.  # needed to accumulate the validation loss

        for i, split in enumerate(cv_splits):
            print 'Cross-validation iteration:', i
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

    def hyperopt_obj_fn_2(self, hyperspace):
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
        num_hids = [int(num_hid) for num_hid in sampled_mln_params['num_hids']]
        activs = list(sampled_mln_params['activs'])
        activs.append('softmax')

        # set the loss terms
        # this hyperspace enforces dropout
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

        return self.compute_cv_loss(mln_params, rmsprop_params)

    def hyperopt_obj_fn(self, hyperspace):
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
        num_hids = [int(num_hid) for num_hid in sampled_mln_params['num_hids']]
        activs = list(sampled_mln_params['activs'])
        activs.append('softmax')

        # set the loss terms
        loss_terms = ['cross_entropy']
        dropout = sampled_mln_params['dropout']
        input_p = None
        hidden_p = None
        l1_decay = sampled_mln_params['l1_reg']
        l2_decay = sampled_mln_params['l2_reg']

        if not dropout is None:
            loss_terms.append('dropout')
            input_p = dropout['input_p']
            hidden_p = dropout['hidden_p']
        if not l1_decay is None:
            loss_terms.append('l1_reg')
        if not l2_decay is None:
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

        return self.compute_cv_loss(mln_params, rmsprop_params)

    def run_hyperopt(self, k_cv=5):
        ''' convenience function for running hyperopt '''

        hyperspace_2 = self.get_hyperspace_2()
        best = fmin(self.hyperopt_obj_fn_2, hyperspace_2, algo=tpe.suggest,
                    max_evals=200)

        return best
