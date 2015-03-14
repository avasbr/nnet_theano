import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
from deepnet.common import nnettrain as nt
from deepnet import MultilayerNet as mln
import theano
import sys


def load_mnist(data_path):
    ''' Loads the MNIST data from the base path '''

    train_img_path = '%s/train-images.idx3-ubyte' % data_path

    # get the training data
    train_img = idx2numpy.convert_from_file(train_img_path)
    m, row, col = train_img.shape
    d = row * col
    X_tr = np.reshape(train_img, (m, d)) / 255.

    return X_tr


def visualize_image_bases(X_max, w=28, h=28):
    plt.figure()
    n_filt = 196
    for i in range(n_filt):
        plt.subplot(14, 14, i)
        curr_img = X_max[i, :].reshape(w, h)
        curr_img /= 1. * np.max(curr_img)  # for consistency
        plt.imshow(curr_img, cmap='gray', interpolation='none')


def main(argv):
    data_path = argv[1]  # directory path that contains the image data
    config_file = argv[2]  # config file which holds all the parameters

    # load data
    print 'Loading data...'
    X_tr = load_mnist(data_path)

    # train a neural network
    print 'Training...'
    sae = nt.train_nnet(config_file, X_tr=X_tr)

    # visualize it
    X_max = sae.compute_max_activations()
    visualize_image_bases(X_max)
    plt.show()

if __name__ == '__main__':
    main(sys.argv)
