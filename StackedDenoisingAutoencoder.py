import theano.tensor as T
import theano
import numpy as np
import MultilayerNet as mln
import Autoencoder as ae
import NeuralNetworkCore

class StackedDenoisingAutoencoder(NeuralNetworkCore.Network):

