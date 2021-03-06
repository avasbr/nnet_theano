# This demo demonstrates how an autoencoder enforcing sparsity can learn edge-filters from
# sampling patches of textured images

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from deepnet import Autoencoder as ae
from deepnet.common import nnetutils as nu
from deepnet.common import nnettrain as nt
import sys


def sample_images(I, w=8, h=8, n=10000):
    '''Extracts n patches (flattened) of size w x h from one of the images in I

    Parameters:
    -----------
    I:  image set
        r x c x i numpy array, r = rows, c = columns, i = # of images

    w:  width of patch
        int
    h:  height of patch
        int

    Returns:
    --------
    X:  data matrix
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
    I:  image set
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


def compute_cv_loss(X, config_file):
    ''' Uses k-fold cross-validation to compute the average loss '''

    k_cv = 5
    cv_splits = nu.split_k_fold_cross_val(X, k_cv=k_cv)
    loss = 0.
    for idx, split in enumerate(cv_splits):
        X_tr, X_val = split
        nnet = nt.train_nnet(config_file, X_tr=X_tr)
        curr_loss = float(nnet.compute_reconstruction_loss(X_val))
        if np.isnan(curr_loss):
            curr_loss = np.inf
        loss += curr_loss

    avg_loss = 1. * loss / k_cv
    return avg_loss


def main(argv):
    data_path = argv[1]  # directory path that contains the image data
    n = int(argv[2])
    config_file = argv[3]  # config file which holds all the parameters

    # load data
    print 'Loading data...'
    I = load_images(data_path)
    X = sample_images(I, n=n)

    print 'Average cross-validation Loss:', compute_cv_loss(X, config_file)

    # train a neural network
    print 'Training...'
    sae = nt.train_nnet(config_file, X_tr=X)

    print 'Training loss:', sae.compute_reconstruction_loss(X)
    # visualize it
    X_max = sae.compute_max_activations()
    visualize_image_bases(X_max, 25)
    plt.show()

if __name__ == '__main__':
    main(sys.argv)
