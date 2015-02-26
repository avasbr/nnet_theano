import numpy as np
from deepnet import NeuralNetworkCore
from deepnet.common import nnetloss as nl
import theano
import theano.tensor as T
import sys


class Autoencoder(NeuralNetworkCore.Network):

  def __init__(self, d=None, num_hids=None, activs=None, tied_wts=False, loss_terms=None, **loss_params):
    ''' implementation of the basic autoencoder '''

    # the autoencoder can only have one hidden layer (and therefore, only two
    # activation functions)
    assert len(num_hids) == 1 and len(activs) == 2

    super(Autoencoder, self).__init__(
        d=d, k=d, num_hids=num_hids, activs=activs, loss_terms=loss_terms, **loss_params)

    # functions that will be available after running the 'fit' method on the
    # autoencoder
    self.decode = None
    self.encode = None
    self.tied_wts = tied_wts

  def set_weights(self, wts=None, bs=None, init_method=None, scale_factor=None, seed=None):
    ''' Initializes the weights and biases of the neural network 

    Parameters:
    -----------
    param: wts - weights
    type: np.ndarray, optional

    param: bs - biases
    type: np.ndarray, optional

    param: method - calls some pre-specified weight initialization routines
    type: string, optional
    '''
    # with tied weights, the encoding and decoding matrices are simply transposes
    # of one another

    if self.tied_wts:
      self.num_nodes = [d] + num_hids

      if seed is not None:
        np.random.seed(seed=seed)

      # weights and biases
      if wts is None and bs is None:
        wts = [None, None]
        bs = [None, None]

        if method == 'gauss':
          wts[0] = scale_factor * np.random.randn(d, num_hids[0])
          # this is a shallow transpose (changing [0] will change this as well)
          wts[1] = wts[0].T
          bs[0] = np.zeros(num_hids[0])
          bs[1] = np.zeros(d)

        if method == 'fan-io':
          v = np.sqrt(1. * scale_factor / (d + num_hids[0] + 1))
          wts[0] = scale_factor * v * np.random.rand(d, num_hids[0] - v)
          wts[1] = wts[0].T
          bs[0] = np.zeros(num_hids[0])
          bs[1] = np.zeros(d)
      else:
        assert isinstance(wts, list)
        assert isinstance(bs, list)

      self.wts_ = [theano.shared(nu.floatX(wt), borrow=True) for wt in wts]
      self.bs_ = [theano.shared(nu.floatX(b), borrow=True) for b in bs]

    # if encoding and decoding matrices are distinct, just default back to the
    # normal case
    else:
      super(Autoencoder, self).set_weights(
          init_method=init_method, scale_factor=scale_factor, seed=seed)

  def corrupt_input(self, X, v=0.1, method='mask'):
    ''' corrupts the input using one of several methods

    Parameters:
    -----------
    param: X - input matrix
    type: np.ndarray

    param: v - either the proportion of values to corrupt, or std for gaussian
    type: float

    param: method - either 'mask', 'gauss'
    type: string
    '''
    if method == 'mask':
      return X * self.srng.binomial(X.shape, n=1, p=1 - v, dtype=theano.config.floatX)
    elif method == 'gauss':
      # additive gaussian noise
      return X + self.srng.normal(X.shape, avg=0.0, std=v, dtype=theano.config.floatX)

  def fit(self, X_tr, X_val=None, wts=None, bs=None, **optim_params):
    ''' calls the fit function of the super class (NeuralNetworkCore) and also compiles the 
    encoding and decoding functions'''

    super(Autoencoder, self).fit(
        X_tr, X_tr, X_val=X_val, y_val=X_val, wts=None, bs=None, **optim_params)
    self.compile_autoencoder_functions()

  def train_fprop(self, X_tr, wts=None, bs=None):
    ''' Performs forward propagation for training, which could be different from
    the vanilla fprop we would use for testing, due to extra bells and whistles such as 
    dropout, corruption, etc'''

    if wts is None and bs is None:
      wts = self.wts_
      bs = self.bs_

    if 'corruption' in self.loss_terms:
      # get the input and hidden layer dropout probabilities
      corrupt_p = self.loss_params['corrupt_p']
      corrupt_type = self.loss_params['corrupt_type']

      self.hidden_act = self.activs[0](T.dot(self.corrupt_input(
          X_tr, corrupt_p, corrupt_type), wts[0]) + bs[0])  # compute the first activation
      return self.activs[1](T.dot(self.hidden_act, wts[1]) + bs[1])
    else:
      return self.fprop(X_tr, wts, bs)

  def fprop(self, X_tr, wts=None, bs=None):
    ''' Performs forward propagation through the network - this fprop is simplified
    specifically for autoencoders, which have only one hidden layer

    Parameters
    ----------
    param: X - training data
    type: theano matrix

    param: wts - weights
    type: theano matrix

    param: bs - biases
    type: theano matrix

    Returns:
    --------
    param: final activation values
    type: theano matrix
    '''
    if wts is None and bs is None:
      wts = self.wts_
      bs = self.bs_

    # this is useful to keep around for when we introduce sparsity
    self.hidden_act = self.activs[0](T.dot(X_tr, wts[0]) + bs[0])
    return self.activs[1](T.dot(self.hidden_act, wts[1]) + bs[1])

  def compute_optim_loss(self, X, y, wts=None, bs=None):
    ''' Given inputs, returns the loss at the current state of the model'''

    if wts is None and bs is None:
      wts = self.wts_
      bs = self.bs_

    # call the super-class function first...
    optim_loss = super(Autoencoder, self).compute_optim_loss(X, y, wts, bs)

    # ... and augment with the sparsity term, if needed
    if 'sparsity' in self.loss_terms:
      beta = self.loss_params.get('beta')
      rho = self.loss_params.get('rho')
      optim_loss += nl.sparsity(self.hidden_act, beta=beta, rho=rho)

    return optim_loss

  def compile_autoencoder_functions(self, wts=None, bs=None):
    ''' compiles the encoding, decoding, and pre-training functions of the autoencoder 

    Parameters
    ----------
    param: wts - weights
    type: theano matrix

    param: bs - biases
    type: theano matrix
    '''

    if wts is None and bs is None:
      wts = self.wts_
      bs = self.bs_

    X = T.matrix()  # features to encode or decode

    # encode the raw data into features defined by the hidden layer
    self.encode = theano.function(
        inputs=[X],
        outputs=self.activs[0](T.dot(X, wts[0]) + bs[0]),
        allow_input_downcast=True)

    # decode the encoded features back into the raw data representation
    self.decode = theano.function(
        inputs=[X],
        outputs=self.activs[1](T.dot(X, wts[1]) + bs[1]),
        allow_input_downcast=True)

    # compute the features that maximally activate each of the features of the
    # hidden layer
    self.compute_max_activations = theano.function(
        inputs=[],
        outputs=wts[0].T /
        (T.sqrt(T.sum(wts[0] ** 2, axis=0)).dimshuffle(0, 'x')),
        allow_input_downcast=True)

    # computes the reconstruction loss from encoding and decoding raw data
    eval_loss = self.compute_eval_loss(X, X, wts, bs)
    self.compute_reconstruction_loss = theano.function(
        inputs=[X],
        outputs=eval_loss,
        allow_input_downcast=True)

    # debugging functions

    # sparsity loss
    # A = T.matrix()
    # b = T.fscalar()
    # r = T.fscalar()
    # self.sparsity_loss = theano.function(
    # 	inputs=[A],
    # 	outputs=[nl.sparsity(A,beta=b,rho=r)],
    # 	allow_input_downcast=True)

  def get_encoding_weights(self):
    ''' get the encoding weights from the shared variable '''
    return self.wts_[0].get_value()
