# Currently very much a work in progress, and contains a lot of hard-coded/repeated code. this was
# done in a hurry though, but ideally, we would have a config file which ingests in "default" vs
# "searchable" parameters, and it will use some kind of val/cross-val loss to
import numpy as np
import matplotlib.pyplot as plt
import theano
import copy
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
        self.curr_X = None
        self.pretrain_wts = []
        self.pretrain_bs = []

    def merge_default_search(self, default_space, search_space):
        merged_space = default_space.copy()
        merged_space.update(search_space)

        return merged_space

    def dict_tuple_to_list(self, d):
        ''' takes dicts and replaces values in tuples with lists '''
        for k, v in d.iteritems():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    def set_finetune_space(self, config_file):
        ''' Given the original deep net architecture, and a set of pretrained weights
        and biases, define the configuration space to search for fintuning parameters '''

        # we know these fields won't change, so go ahead and set them as
        # defaults now
        model_params = nt.get_model_params(config_file)
        optim_params = nt.get_optim_params(config_file)
        default_finetune_model_params = {k: model_params[k] for k in ('num_hids', 'activs', 'd', 'k')}
        default_finetune_model_params['loss_terms'] = ['cross_entropy']
        default_finetune_optim_params = {k: optim_params[k] for k in ('optim_method', 'optim_type')}

        # define the space of hyperparameters we wish to
        search_finetune_model_params = {'l1_reg': hp.choice('l1_reg', [None, hp.loguniform('l1_decay', log(1e-5), log(10))]),
                                        'l2_reg': hp.choice('l2_reg', [None, hp.loguniform('l2_decay', log(1e-5), log(10))])}
        search_finetune_optim_params = {'learn_rate': hp.uniform('learn_rate', 0, 1),
                                        'rho': hp.uniform('rho', 0, 1),
                                        'num_epochs': hp.qloguniform('num_epochs', log(10), log(5e3), 1),
                                        'batch_size': hp.quniform('batch_size', 128, 1024, 1),
                                        'init_method': hp.choice('init_method', ['gauss', 'fan-io']),
                                        'scale_factor': hp.uniform('scale_factor', 0, 1)}

        # combine the default and search parameters into a dictionary to define the
        # full space - this is what will be passed into the objective function
        finetune_model_params = self.merge_default_search(
            default_finetune_model_params, search_finetune_model_params)
        finetune_optim_params = self.merge_default_search(
            default_finetune_optim_params, search_finetune_optim_params)

        finetune_hyperspace = {
            'finetune_model_params': finetune_model_params, 'finetune_optim_params': finetune_optim_params}

        return finetune_hyperspace

    def compute_finetune_objective(self, hyperspace):
        ''' objective function for finetuning '''

        curr_model_params = {k: hyperspace['finetune_model_params'][k] for k in ('num_hids', 'activs', 'd',
                                                                                 'k', 'loss_terms')}
        curr_optim_params = {k: hyperspace['finetune_optim_params'][k] for k in ('optim_type', 'optim_method',
                                                                                 'learn_rate', 'rho', 'num_epochs',
                                                                                 'batch_size', 'scale_factor')}
        # clean up
        curr_model_params = self.dict_tuple_to_list(
            curr_model_params)
        curr_optim_params['num_epochs'] = int(
            curr_optim_params['num_epochs'])
        curr_optim_params['batch_size'] = int(
            curr_optim_params['batch_size'])

        # there's a little extra we need to do before this is completely ready
        if 'l1_decay' in hyperspace['finetune_model_params']:
            curr_model_params['loss_terms'].append('l1_reg')
            curr_model_params[
                'l1_decay':hyperspace['finetune_model_params']['l1_decay']]

        if 'l2_decay' in hyperspace['finetune_model_params']:
            curr_model_params['loss_terms'].append('l2_reg')
            curr_model_params[
                'l2_decay':hyperspace['finetune_model_params']['l2_decay']]

        if hyperspace['finetune_optim_params']['init_method'] == 0:
            curr_optim_params['init_method'] = 'gauss'
        else:
            curr_optim_params['init_method'] = 'fan-io'

        curr_optim_params['num_epochs'] = int(curr_optim_params['num_epochs'])
        curr_optim_params['batch_size'] = int(curr_optim_params['batch_size'])
        return self.compute_val_loss(curr_model_params, curr_optim_params,
                                     wts=self.pretrain_wts, bs=self.pretrain_bs)

    def learn_pretrain_settings(self, config_file):
        ''' automatically searches for the right settings to pre-train the model described in the configuration path '''

        # seperate out the default from the search space
        default_pretrain_model_params = {'corrupt_type': 'mask', 'activs': ['sigmoid', 'sigmoid'],
                                         'loss_terms': ['binary_cross_entropy', 'corruption']}

        default_pretrain_optim_params = {
            'optim_type': 'minibatch', 'optim_method': 'RMSPROP'}

        default_last_model_params = {
            'activs': ['softmax'], 'loss_terms': ['cross_entropy'], 'num_hids': []}

        default_last_optim_params = {
            'optim_type': 'minibatch', 'optim_method': 'RMSPROP'}

        # get the number of nodes per hidden layer from the original architecture so we know how to
        # initialize the autoencoders and the final softmax layer
        model_params = nt.get_model_params(config_file)
        pretrain_nodes = [model_params['d']] + model_params['num_hids']

        best_pretrain_settings = []
        self.curr_X = self.X

        print 'Starting pre-training...'

        for l1, l2 in zip(pretrain_nodes[:-1], pretrain_nodes[1:]):

             # we will reuse this hyperspace (denoising autoencoder) for every pair
            # of input-output layers
            search_pretrain_model_params = {'corrupt_p': hp.uniform('corrupt_p', 0, 1),
                                            'l1_reg': hp.choice('l1_reg', [None, hp.loguniform('l1_decay', log(1e-5), log(10))]),
                                            'l2_reg': hp.choice('l2_reg', [None, hp.loguniform('l2_decay', log(1e-5), log(10))])}

            search_pretrain_optim_params = {'learn_rate': hp.uniform('learn_rate', 0, 1),
                                            'rho': hp.uniform('rho', 0, 1),
                                            'num_epochs': hp.qloguniform('num_epochs', log(1e2), log(2000), 1),
                                            'batch_size': hp.quniform('batch_size', 128, 1024, 1),
                                            'init_method': hp.choice('init_method', ['gauss', 'fan-io']),
                                            'scale_factor': hp.uniform('scale_factor', 0, 1)}

            # get the next pretraining space ready
            default_pretrain_model_params['d'] = l1
            default_pretrain_model_params['num_hids'] = [l2]

            curr_pretrain_model_params = self.merge_default_search(
                default_pretrain_model_params, search_pretrain_model_params)

            curr_pretrain_optim_params = self.merge_default_search(
                default_pretrain_optim_params, search_pretrain_optim_params)

            # combine the merged default model and optim parameters into a
            # single dictionary
            pretrain_hyperspace = {'pretrain_model_params': curr_pretrain_model_params,
                                   'pretrain_optim_params': curr_pretrain_optim_params}

            # search over the hyperparameters
            print 'Searching over the pretraining hyperspace...'
            best = fmin(self.compute_pretrain_objective, pretrain_hyperspace, algo=tpe.suggest,
                        max_evals=100)
            print 'Complete!'
            print 'Training this layer'

            best_pretrain_settings.append(best)

            # this updates curr_X for the next pre-training layer, and also stores the pre-trained
            # weights
            self.pretrain_layer_with_settings(
                best,
                search_pretrain_model_params,
                search_pretrain_optim_params,
                default_pretrain_model_params,
                default_pretrain_optim_params, layer_type='pretrain')

        print 'Pretraining the final layer..'
        # the last layer is not pretrained with an autoencoder...
        default_last_model_params['d'] = l2
        default_last_model_params['k'] = model_params['k']

        # ...it is trained normally, via simple softmax regression
        search_last_model_params = {'l1_reg': hp.choice('l1_reg', [None, hp.loguniform('l1_decay', log(1e-5), log(10))]),
                                    'l2_reg': hp.choice('l2_reg', [None, hp.loguniform('l2_decay', log(1e-5), log(10))])}
        search_last_optim_params = {'learn_rate': hp.uniform('learn_rate', 0, 1),
                                    'rho': hp.uniform('rho', 0, 1),
                                    'num_epochs': hp.qloguniform('num_epochs', log(1e2), log(2000), 1),
                                    'batch_size': hp.quniform('batch_size', 128, 1024, 1),
                                    'init_method': hp.choice('init_method', ['gauss', 'fan-io']),
                                    'scale_factor': hp.uniform('scale_factor', 0, 1)}

        last_model_params = self.merge_default_search(
            default_last_model_params, search_last_model_params)

        last_optim_params = self.merge_default_search(
            default_last_optim_params, search_last_optim_params)

        last_hyperspace = {
            'last_model_params': last_model_params, 'last_optim_params': last_optim_params}
        best = fmin(self.compute_last_objective, last_hyperspace, algo=tpe.suggest,
                    max_evals=1)
        best_pretrain_settings.append(best)

        self.pretrain_layer_with_settings(
            best,
            search_last_model_params,
            search_last_optim_params,
            default_last_model_params,
            default_last_optim_params, layer_type='last')

        print 'Complete!'
        return best_pretrain_settings

    def compute_pretrain_objective(self, hyperspace):
        ''' objective function for pre-training layers '''

        # parse the hyperparams from the curr hyperspace
        curr_pretrain_model_params = {k: hyperspace['pretrain_model_params'][k] for k in ('d', 'num_hids', 'corrupt_p', 'loss_terms',
                                                                                          'corrupt_type', 'activs')}
        curr_pretrain_optim_params = {k: hyperspace['pretrain_optim_params'][k] for k in ('optim_method', 'optim_type', 'learn_rate', 'rho',
                                                                                          'num_epochs', 'batch_size', 'scale_factor')}

        # hyperopt decides to change lists to tuples, so.. change them back.. not the cleanest way to
        # handle this, but whatever, for now

        curr_pretrain_model_params = self.dict_tuple_to_list(
            curr_pretrain_model_params)
        curr_pretrain_optim_params['num_epochs'] = int(
            curr_pretrain_optim_params['num_epochs'])
        curr_pretrain_optim_params['batch_size'] = int(
            curr_pretrain_optim_params['batch_size'])

        # there's a little extra we need to do before this is completely ready
        if 'l1_decay' in hyperspace['pretrain_model_params']:
            curr_pretrain_model_params['loss_terms'].append('l1_reg')
            curr_pretrain_model_params[
                'l1_decay': hyperspace['pretrain_model_params']['l1_decay']]

        if 'l2_decay' in hyperspace['pretrain_model_params']:
            curr_pretrain_model_params['loss_terms'].append('l2_reg')
            curr_pretrain_model_params[
                'l2_decay': hyperspace['pretrain_model_params']['l2_decay']]

        if hyperspace['pretrain_optim_params']['init_method'] == 0:
            curr_pretrain_optim_params['init_method'] = 'gauss'
        else:
            curr_pretrain_optim_params['init_method'] = 'fan-io'

        print 'Pretraining parameters'
        print curr_pretrain_model_params
        print 'Optimization parameters'
        print curr_pretrain_optim_params

        return self.compute_val_reconstruction_loss(curr_pretrain_model_params, curr_pretrain_optim_params)

    def compute_last_objective(self, hyperspace):

        curr_last_model_params = {k: hyperspace['last_model_params'][k] for k in ('d', 'k', 'num_hids', 'loss_terms', 'activs')}
        curr_last_optim_params = {k: hyperspace['last_optim_params'][k] for k in ('optim_method', 'optim_type', 'learn_rate', 'rho',
                                                                                  'num_epochs', 'batch_size', 'scale_factor')}
        # hyperopt decides to change lists to tuples, so.. change them back.. not the cleanest way to
        # handle this, but whatever, for now
        curr_last_model_params = self.dict_tuple_to_list(
            curr_last_model_params)
        curr_last_optim_params['num_epochs'] = int(
            curr_last_optim_params['num_epochs'])
        curr_last_optim_params['batch_size'] = int(
            curr_last_optim_params['batch_size'])

        # there's a little extra we need to do before this is completely ready
        if 'l1_decay' in hyperspace['last_model_params']:
            curr_last_model_params['loss_terms'].append('l1_reg')
            curr_last_model_params[
                'l1_decay': hyperspace['last_model_params']['l1_decay']]

        if 'l2_decay' in hyperspace['last_model_params']:
            curr_last_model_params['loss_terms'].append('l2_reg')
            curr_last_model_params[
                'l2_decay': hyperspace['last_model_params']['l2_decay']]

        if hyperspace['last_optim_params']['init_method'] == 0:
            curr_last_optim_params['init_method'] = 'gauss'
        else:
            curr_last_optim_params['init_method'] = 'fan-io'

        print 'Last parameters'
        print curr_last_model_params
        print 'Optimization parameters'
        print curr_last_optim_params

        return self.compute_val_loss(curr_last_model_params, curr_last_optim_params, X=self.curr_X, y=self.y)

    def pretrain_layer_with_settings(self, best, search_model_params, search_optim_params, default_model_params, default_optim_params, layer_type):
        ''' given a learned best-setting and the default model and optimization parameters, pretrain the weights
        for that layer '''

        model_params = copy.deepcopy(default_model_params)
        optim_params = copy.deepcopy(default_optim_params)

        # these are the keys we want to search in 'best'
        search_model_keys = [k for k in search_model_params]
        search_optim_keys = [k for k in search_optim_params]

        for k in search_model_keys:
            if k == 'l1_reg' and 'l1_decay' in best:
                model_params['loss_terms'].append('l1_reg')
                model_params['l1_decay'] = best['l1_decay']
            elif k == 'l2_reg' and 'l2_decay' in best:
                model_params['loss_terms'].append('l2_reg')
                model_params['l2_decay'] = best['l2_decay']
            else:
                model_params[k] = best[k]

        for k in search_optim_keys:
            if k == 'init_method':
                if best[k] == 0:
                    optim_params[k] = 'gauss'
                else:
                    optim_params[k] = 'fan-io'
            elif k == 'num_epochs' or k == 'batch_size':
                optim_params[k] = int(best[k])
            else:
                optim_params[k] = best[k]

        # pre-train the layer
        if layer_type == 'pretrain':
            nnet = ae.Autoencoder(**model_params)
            nnet.fit(self.curr_X, **optim_params)

            # we'll need these to initialize the final net for fine-tuning
            self.pretrain_wts.append(nnet.wts_[0].get_value())
            self.pretrain_bs.append(nnet.bs_[0].get_value())

            # and set the input to the next layer
            self.curr_X = nnet.encode(self.curr_X)

        elif layer_type == 'last':
            nnet = mln.MultilayerNet(**model_params)
            nnet.fit(self.curr_X, self.y, **optim_params)
            self.pretrain_wts.append(nnet.wts_[0].get_value())
            self.pretrain_bs.append(nnet.bs_[0].get_value())
        else:
            print sys.exit('Invalid layer type; only "pretrain" and "last" are allowed')

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

        default_mln_model_params = {
            'd': self.d, 'k': self.k, 'loss_terms': ['cross_entropy', 'dropout']}

        search_mln_model_params = {
            'arch': hp.choice('arch', nnets),
            'input_p': hp.uniform('ip', 0, 1),
            'hidden_p': hp.uniform('hp', 0, 1),
            'l1_reg': hp.choice('l1_reg', [None, hp.loguniform('l1_decay', log(1e-5), log(10))]),
            'l2_reg': hp.choice('l2_reg', [None, hp.loguniform('l2_decay', log(1e-5), log(10))])}

        default_mln_optim_params = {
            'optim_type': 'minibatch', 'optim_method': 'RMSPROP'}

        search_mln_optim_params = {
            'learn_rate': hp.uniform('learn_rate', 0, 1),
            'rho': hp.uniform('rho', 0, 1),
            'num_epochs': hp.qloguniform('num_epochs', log(1e2), log(2000), 1),
            'batch_size': hp.quniform('batch_size', 128, 1024, 1),
            'init_method': hp.choice('init_method', ['gauss', 'fan-io']),
            'scale_factor': hp.uniform('scale_factor', 0, 1)}

        # merge the default and search spaces
        mln_model_params = self.merge_default_search(
            default_mln_model_params, search_mln_model_params)
        mln_optim_params = self.merge_default_search(
            default_mln_optim_params, search_mln_optim_params)

        # define the hyperparamater space to search
        hyperspace = {'mln_model_params': mln_model_params,
                      'mln_optim_params': mln_optim_params}

        return hyperspace

    def compute_multilayer_dropout_objective(self, hyperspace):
        ''' parses the multilayer with dropout hyperspace and translates it into a loss value which
        we will use to search the space of hyperparams '''

        curr_model_params = {k: hyperspace['mln_model_params'][k] for k in ('d', 'k', 'loss_terms', 'input_p', 'hidden_p')}
        curr_optim_params = {k: hyperspace['mln_optim_params'][k] for k in ('optim_type', 'optim_method',
                                                                            'learn_rate', 'rho', 'num_epochs',
                                                                            'batch_size', 'scale_factor')}
        # clean up
        curr_model_params = self.dict_tuple_to_list(
            curr_model_params)
        curr_optim_params['num_epochs'] = int(
            curr_optim_params['num_epochs'])
        curr_optim_params['batch_size'] = int(
            curr_optim_params['batch_size'])

        # there's a little extra we need to do before this is completely ready
        if 'l1_decay' in hyperspace['mln_model_params']:
            curr_model_params['loss_terms'].append('l1_reg')
            curr_model_params[
                'l1_decay':hyperspace['mln_model_params']['l1_decay']]

        if 'l2_decay' in hyperspace['mln_model_params']:
            curr_model_params['loss_terms'].append('l2_reg')
            curr_model_params[
                'l2_decay':hyperspace['mln_model_params']['l2_decay']]

        if hyperspace['mln_optim_params']['init_method'] == 0:
            curr_optim_params['init_method'] = 'gauss'
        else:
            curr_optim_params['init_method'] = 'fan-io'

        # collect number of hidden units and define activation functions
        num_hids = list(hyperspace['mln_model_params']['arch'])
        activs = ['reLU'] * len(num_hids) + ['softmax']

        curr_model_params['num_hids'] = num_hids
        curr_model_params['activs'] = activs

        print 'Multilayer net parameters'
        print curr_model_params
        print 'Optimization parameters'
        print curr_optim_params

        return self.compute_val_loss(curr_model_params, curr_optim_params)
        # return self.compute_cv_loss(mln_model_params, rmsprop_params)

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

        default_mln_model_params = {
            'd': self.d, 'k': self.k, 'loss_terms': ['cross_entropy']}

        search_mln_model_params = {
            'arch': hp.choice('arch', nnets),
            'l1_reg': hp.choice('l1_reg', [None, hp.loguniform('l1_decay', log(1e-5), log(10))]),
            'l2_reg': hp.choice('l2_reg', [None, hp.loguniform('l2_decay', log(1e-5), log(10))])}

        default_mln_optim_params = {
            'optim_type': 'minibatch', 'optim_method': 'RMSPROP'}

        search_mln_optim_params = {
            'learn_rate': hp.uniform('learn_rate', 0, 1),
            'rho': hp.uniform('rho', 0, 1),
            'num_epochs': hp.qloguniform('num_epochs', log(1e2), log(2000), 1),
            'batch_size': hp.quniform('batch_size', 128, 1024, 1),
            'init_method': hp.choice('init_method', ['gauss', 'fan-io']),
            'scale_factor': hp.uniform('scale_factor', 0, 1)}

        # merge the default and search spaces
        mln_model_params = self.merge_default_search(
            default_mln_model_params, search_mln_model_params)
        mln_optim_params = self.merge_default_search(
            default_mln_optim_params, search_mln_optim_params)

        # define the hyperparamater space to search
        hyperspace = {'mln_model_params': mln_model_params,
                      'mln_optim_params': mln_optim_params}

        return hyperspace

    def compute_old_objective(self, hyperspace):
        ''' objective function that takes in a hyperspace and returns a cost/value '''

        curr_model_params = {k: hyperspace['mln_model_params'][k] for k in ('d', 'k', 'loss_terms')}
        curr_optim_params = {k: hyperspace['mln_optim_params'][k] for k in ('optim_type', 'optim_method',
                                                                            'learn_rate', 'rho', 'num_epochs',
                                                                            'batch_size', 'scale_factor')}
        # clean up
        curr_model_params = self.dict_tuple_to_list(
            curr_model_params)
        curr_optim_params['num_epochs'] = int(
            curr_optim_params['num_epochs'])
        curr_optim_params['batch_size'] = int(
            curr_optim_params['batch_size'])

        # there's a little extra we need to do before this is completely ready
        if 'l1_decay' in hyperspace['mln_model_params']:
            curr_model_params['loss_terms'].append('l1_reg')
            curr_model_params[
                'l1_decay':hyperspace['mln_model_params']['l1_decay']]

        if 'l2_decay' in hyperspace['mln_model_params']:
            curr_model_params['loss_terms'].append('l2_reg')
            curr_model_params[
                'l2_decay':hyperspace['mln_model_params']['l2_decay']]

        if hyperspace['mln_optim_params']['init_method'] == 0:
            curr_optim_params['init_method'] = 'gauss'
        else:
            curr_optim_params['init_method'] = 'fan-io'

        # collect number of hidden units and define activation functions
        num_hids = list(hyperspace['mln_model_params']['arch'])
        activs = ['sigmoid'] * len(num_hids) + ['softmax']

        curr_model_params['num_hids'] = num_hids
        curr_model_params['activs'] = activs

        print 'Multilayer net parameters'
        print curr_model_params
        print 'Optimization parameters'
        print curr_optim_params

        return self.compute_val_loss(curr_model_params, curr_optim_params)

    #-------------------Functions for computing validation---------------

    def compute_val_reconstruction_loss(self, pretrain_params, mln_optim_params, p=0.8):
        ''' Reconstruction loss '''

        X_tr, X_val = nu.split_train_val(self.curr_X, p)
        nnet = ae.Autoencoder(**pretrain_params)
        nnet.fit(X_tr, **mln_optim_params)

        re_val_loss = float(nnet.compute_reconstruction_loss(X_val))
        print 'Reconstruction loss on Validation set:', re_val_loss

        return re_val_loss

    def compute_val_loss(self, mln_model_params, mln_optim_params, X=None, y=None, p=0.8, wts=None, bs=None):
        ''' Uses a single train/val split to compute the loss '''

        if X is None:
            X = self.X
        if y is None:
            y = self.y

        X_tr, y_tr, X_val, y_val = nu.split_train_val(X, p, y=y)
        nnet = mln.MultilayerNet(**mln_model_params)

        # adding optional weights/biases here allows for fine-tuning, if needed
        nnet.fit(X_tr, y_tr, wts=wts, bs=bs, **mln_optim_params)
        val_loss = float(nnet.compute_test_loss(X_val, y_val))

        print 'Validation loss:', val_loss

        return val_loss

    def compute_cv_loss(self, mln_model_params, mln_optim_params, k_cv=5):
        ''' Uses k-fold cross-val to compute the average loss '''

        # get the indices of the splits
        cv_splits = nu.split_k_fold_cross_val(self.X, k_cv=k_cv, y=self.y)

        val_loss = 0.  # needed to accumulate the validation loss

        for i, split in enumerate(cv_splits):
            print 'Cross-validation iteration:', i + 1
            # get the training and validation for this split
            X_tr, y_tr, X_val, y_val = split
            # initialize the neural network
            nnet = mln.MultilayerNet(**mln_model_params)
            nnet.fit(X_tr, y_tr, **mln_optim_params)  # fit to the training
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
        elif self.space_type == 'pretrain-finetune':
            if config_file is None:
                sys.exit(
                    'Cannot pre-train & fine-tune a network without its original config file')
            else:
                pretrain_best = self.learn_pretrain_settings(config_file)
                hyperspace = self.set_finetune_space(config_file)
                finetune_best = fmin(self.compute_finetune_objective, hyperspace, algo=tpe.suggest,
                                     max_evals=1)

                return pretrain_best, finetune_best

        else:
            sys.exit(
                'Space type not specified correctly, your choices are: "modern","old", or "pretrain-finetune"')

        return best
