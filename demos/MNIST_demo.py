import idx2numpy
import numpy as np
from deepnet.common import nnettrain as nt
from deepnet import MultilayerNet as mln
import theano
import sys


def load_mnist(data_path):
    ''' Loads the MNIST data from the base path '''

    train_img_path = '%s/train-images.idx3-ubyte' % data_path
    train_lbl_path = '%s/train-labels.idx1-ubyte' % data_path
    test_img_path = '%s/t10k-images.idx3-ubyte' % data_path
    test_lbl_path = '%s/t10k-labels.idx1-ubyte' % data_path

    def encode_one_hot(y, m, k):
        y_one_hot = np.zeros((m, k))
        y_one_hot[range(m), y] = 1
        return y_one_hot

    # get the training data
    train_img = idx2numpy.convert_from_file(train_img_path)
    m, row, col = train_img.shape
    d = row * col
    X_tr = np.reshape(train_img, (m, d)) / 255.

    train_lbl = idx2numpy.convert_from_file(train_lbl_path)
    k = max(train_lbl) + 1
    y_tr = encode_one_hot(train_lbl, m, k)

    # set the data matrix for test
    test_img = idx2numpy.convert_from_file(test_img_path)
    m_te = test_img.shape[0]
    X_te = np.reshape(test_img, (m_te, d)) / 255.  # test data matrix
    test_lbl = idx2numpy.convert_from_file(test_lbl_path)
    y_te = encode_one_hot(test_lbl, m_te, k)

    return X_tr, y_tr, X_te, y_te


def train_nnet_mnist():
    ''' Trains a neural network on the MNIST data '''

    print 'Loading data...'

    # default paths
    data_path = '/home/avasbr/datasets/MNIST'
    config_file = '/home/avasbr/Desktop/nnet_theano/MNIST_config.ini'
    X_tr, y_tr, X_te, y_te = load_mnist(data_path)

    # Train a model with the same learning rate on the training set, test on
    # the testing set:
    print 'Training...'
    d = X_tr.shape[1]
    k = y_tr.shape[1]

    mln_params = {'d': d, 'k': k, 'num_hids': [1024, 1024], 'activs': ['reLU', 'reLU', 'softmax'],
                  'loss_terms': ['cross_entropy', 'dropout'], 'input_p': 0.2, 'hidden_p': 0.5}

    rmsprop_params = {'init_method': 'gauss', 'scale_factor': 0.01, 'optim_method': 'RMSPROP',
                      'optim_type': 'minibatch', 'num_epochs': 100, 'batch_size': 128, 'learn_rate': 0.001,
                      'rho': 0.9, 'max_norm': True, 'c': 15}

    nnet = mln.MultilayerNet(**mln_params)
    nnet.fit(X_tr, y_tr, **rmsprop_params)

    print 'Performance on test set:'
    print 100 * nnet.score(X_te, y_te), '%'


def main(argv):

    data_path = argv[1]  # directory path that contains all the MNIST data
    config_file = argv[2]  # config file which holds all the parameters

    # load data
    print 'Loading data...'
    X_tr, y_tr, X_te, y_te = load_mnist(data_path)

    # train a neural network
    print 'Training...'
    nnet = nt.train_nnet(config_file, X_tr, y_tr=y_tr)

    # test it
    print 'Performance on test set:'
    print 100 * nnet.score(X_te, y_te), '%'

    # train_nnet_mnist() # runs

if __name__ == '__main__':
    main(sys.argv)
