# This demo demonstrates how an autoencoder enforcing sparsity can learn edge-filters from
# sampling patches of textured images. We also attempt to use hyperopt to decide some of the
# hyperparameters - Andrew Ng's example came with the right hyperparameter settings to use,
# but if one did not know these parameters a priori, we would not be able to produce these results,
# so this is an exercise in seeing how well these parameters can be discovered, or, if there's an
# alternate setting which produces similar results

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from math import log
from hyperopt import hp, fmin, tpe
from hyperopt.pyll.stochastic import sample

from deepnet import Autoencoder as ae
from deepnet.common import nnetutils as nu
from deepnet.common import nnettrain as nt
import sys


def sample_images(I, w=8, h=8, n=10000):
    '''Extracts n patches (flattened) of size w x h from one of the images in I

    Parameters:
    -----------
    I:	image set
            r x c x i numpy array, r = rows, c = columns, i = # of images

    w:	width of patch
            int
    h:	height of patch
            int

    Returns:
    --------
    X:	data matrix
            w*h x n numpy array
    '''
    row, col, idx = I.shape

    # random r_idx,c_idx pairs
    r_idx = np.random.randint(row - h - 1, size=n)
    c_idx = np.random.randint(col - h - 1, size=n)
    X = np.empty((n, w * h))  # empty data matrix

    # for each r,c, pair, extract a patch from a random image,
    # then flatten
    for i, (r, c) in enumerate(zip(r_idx, c_idx)):
        X[i, :] = I[r:r + w, c:c + h, np.random.randint(idx)].flatten()

    X -= np.mean(X, axis=1)[:, np.newaxis]  # zero-mean

    # truncate values to +/- 3 standard deviations and scale to [-1,1]
    pstd = 3 * np.std(X)
    X = np.maximum(np.minimum(X, pstd), -1. * pstd) / pstd

    # rescale to [0.1,0.9]
    X = 0.4 * (X + 1) + 0.1
    return X


def load_images(data_path):
    '''Loads the images from the mat file

    Parameters:
    -----------
    param: data_path - directory for IMAGES.mat
    type: string 

    Returns
    --------
    I:	image set
            r x c x i numpy array, r = rows, c = columns, i = # of images

    '''
    mat = scipy.io.loadmat('%s/IMAGES.mat' % data_path)
    return mat['IMAGES']


def visualize_image_bases(X_max, n_hid, w=8, h=8):
    plt.figure()
    for i in range(n_hid):
        plt.subplot(5, 5, i)
        curr_img = X_max[i, :].reshape(w, h)
        plt.imshow(curr_img, cmap='gray', interpolation='none')


def show_reconstruction(X, X_r, idx, w=8, h=8):
    ''' Plots a single patch before and after reconstruction '''

    plt.figure()
    xo = X[:, idx].reshape(w, h)
    xr = X_r[:, idx].reshape(w, h)
    plt.subplot(211)
    plt.imshow(xo, cmap='gray', interpolation='none')
    plt.title('Original patch')
    plt.subplot(212)
    plt.imshow(xr, cmap='gray', interpolation='none')
    plt.title('Reconstructed patch')

#------------Hyperopt---------------

data_path = '/home/bhargav/datasets/image_patches'  # path to the data
n = 10000  # number of patches to sample
I = load_images(data_path)
X = sample_images(I, n=n)
d = X.shape[1]


def compute_cv_loss(sae_params, optim_params):
    ''' Uses k-fold cross-validation to compute the average loss '''

    k_cv = 5
    cv_splits = nu.split_k_fold_cross_val(X, k_cv=k_cv)
    loss = 0.
    for idx, split in enumerate(cv_splits):
        X_tr, X_val = split
        nnet = ae.Autoencoder(**sae_params)
        nnet.fit(X_tr, **optim_params)
        curr_loss = float(nnet.compute_reconstruction_loss(X_val))
        if np.isnan(curr_loss):
            curr_loss = np.inf
        loss += curr_loss
        print 'Cross validation iteration ', idx, ' Reconstruction loss:', curr_loss

    avg_loss = 1. * loss / k_cv
    print 'Average cross-validation Loss:', avg_loss
    return avg_loss


def hyperopt_obj_fn(hyperspace):

    # parse the hyperparams from the sampled hyperspace
    sampled_ae_params = {}
    sampled_optim_params = {}

    # collect dictionaries into single dictionary
    for param in hyperspace['ae_params']:
        sampled_ae_params.update(param)

    for param in hyperspace['optim_params']:
        if 'batch_size' in param:
            param['batch_size'] = int(param['batch_size'])
        if 'num_epochs' in param:
            param['num_epochs'] = int(param['num_epochs'])
        sampled_optim_params.update(param)

    # set the loss terms
    loss_terms = ['squared_error', 'sparsity']  # enforce sparsity
    l1_decay = sampled_ae_params['l1_reg']
    l2_decay = sampled_ae_params['l2_reg']
    beta = sampled_ae_params['beta']
    sparse_rho = sampled_ae_params['sparse_rho']

    if not l1_decay is None:
        loss_terms.append('l1_reg')
    if not l2_decay is None:
        loss_terms.append('l2_reg')

    # sparse autoencoder parameters
    ae_params = {'d': d, 'num_hids': [25], 'activs': ['sigmoid', 'sigmoid'],
                 'loss_terms': loss_terms, 'l2_decay': l2_decay, 'l1_decay': l1_decay,
                 'beta': beta, 'rho': sparse_rho}

    # rmsprop parameters
    rmsprop_params = {'optim_method': 'RMSPROP', 'optim_type': 'minibatch'}
    rmsprop_params.update(sampled_optim_params)

    # lbfgsb parameters
    lbfgs_params = {'optim_method': 'L-BFGS-B', 'optim_type': 'fullbatch'}
    lbfgs_params.update(sampled_optim_params)

    print 'Sparse Autoencoder parameters'
    print ae_params
    print 'Optim parameters'
    print lbfgs_params

    return compute_cv_loss(ae_params, lbfgs_params)

# define the hyperparamater space to search
# hyperspace = {'ae_params': [
#     {'l1_reg': hp.choice(
#         'l1_lambda', [None, hp.loguniform('l1_decay', log(1e-5), log(10))])},
#     {'l2_reg': hp.choice(
#         'l2_lambda', [None, hp.loguniform('l2_decay', log(1e-5), log(10))])},
#     {'beta': hp.loguniform(
#         'beta', log(1e-2), log(10))},
#     {'sparse_rho': hp.loguniform(
#         'sparse_rho', log(1e-2), log(1))}
# ],
#     'optim_params': [
#     {'learn_rate': hp.uniform('lr', 0, 1)},
#     {'rho': hp.uniform('rho', 0, 1)},
#     {'num_epochs': hp.qloguniform('epochs', log(10), log(1000), 1)},
#     {'batch_size': hp.quniform('size', 128, 1024, 1)},
#     {'init_method': hp.choice(
#         'init', ['gauss', 'fan-io'])},
#     {'scale_factor': hp.uniform('scale', 0, 1)}
# ]
# }

hyperspace = {'ae_params': [
    {'l1_reg': hp.choice(
        'l1_lambda', [None, hp.loguniform('l1_decay', log(1e-5), log(10))])},
    {'l2_reg': hp.choice(
        'l2_lambda', [None, hp.loguniform('l2_decay', log(1e-5), log(10))])},
    {'beta': hp.loguniform(
        'beta', log(1e-2), log(10))},
    {'sparse_rho': hp.loguniform(
        'sparse_rho', log(1e-2), log(1))}
],
    'optim_params': [
    {'num_epochs': hp.qloguniform('epochs', log(10), log(1000), 1)},
    {'batch_size': hp.quniform('size', 128, 1024, 1)},
    {'init_method': hp.choice(
        'init', ['gauss', 'fan-io'])},
    {'scale_factor': hp.uniform('scale', 0, 1)}
]
}

print 'Running hyperopt to determine good hyperparameter settings'
best = fmin(hyperopt_obj_fn, hyperspace, algo=tpe.suggest, max_evals=100)
print best

# print 'Trying out some failed hyperparameters'

# ae_terms = {'beta': 0.5445760025914531, 'd': 64, 'rho': 0.01623096175544125, 'num_hids': [25],
#             'l1_decay': None, 'loss_terms': ['squared_error', 'sparsity', 'l2_reg'],
#             'l2_decay': 0.0004294577310202803, 'activs': ['sigmoid', 'sigmoid']}

# opt_terms = {'learn_rate': 0.7477276197510565, 'scale_factor': 0.08740630845952124,
#              'optim_type': 'minibatch', 'batch_size': 243, 'optim_method': 'RMSPROP',
#              'init_method': 'fan-io', 'rho': 0.11547856267104184, 'num_epochs': 80}

# sparse autoencoder parameters
# ae_params = {'d': d, 'num_hids': [25], 'activs': ['sigmoid', 'sigmoid'],
#              'loss_terms': ['squared_error', 'l2_reg', 'sparsity'], 'beta': ae_terms['beta'],
#              'rho': ae_terms['rho'], 'l2_decay':ae_terms['l2_decay']}

# rmsprop parameters
# rmsprop_params = {'optim_method': 'RMSPROP', 'optim_type': 'minibatch',
#                   'learn_rate': opt_terms['learn_rate'], 'init_method': opt_terms['init_method'],
#                   'scale_factor': opt_terms['scale_factor'],'rho': opt_terms['rho'], 'num_epochs': opt_terms['num_epochs'],
#                   'batch_size': opt_terms['batch_size']}

# compute_cv_loss(ae_params, rmsprop_params)
