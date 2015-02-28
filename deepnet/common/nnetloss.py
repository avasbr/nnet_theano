import theano
import theano.tensor as T
import numpy as np


def l1_reg(wts, l1_decay=None):
    ''' l1 regularization

    Parameters:
    -----------
    param: wts - weights
    type: theano shared matrix

    param: l1_decay - l1 decay term
    type: float

    Returns:
    --------
    param: reg_loss - regularization loss term
    type: theano scalar
    '''

    reg_loss = 0
    # if l1_decay is None, or some wise-guy sets it to 0.0, no need to compute
    # it
    if l1_decay is not None or not l1_decay == 0.0:
        reg_loss += l1_decay * sum([T.sum(T.abs_(w)) for w in wts])
    return reg_loss


def l2_reg(wts, l2_decay=None):
    ''' l2 regularization

    Parameters:
    -----------
    param: wts - weights
    type: theano shared matrix

    param: l2_decay - l2 decay term
    type: float

    Returns:
    --------
    param: reg_loss - regularization loss term
    type: theano scalar
    '''
    reg_loss = 0
    # if l1_decay is None, or some wise-guy sets it to 0.0, no need to compute
    # it
    if l2_decay is not None or not l2_decay == 0.0:
        reg_loss += l2_decay * sum([T.sum(w ** 2) for w in wts])
    return reg_loss


def cross_entropy(y, y_pred):
    ''' Basic cross entropy loss function

    Parameters:
    -----------
    param: y - true labels
    type: theano matrix

    param: y_pred - predicted labels
    type: theano matrix

    Returns:
    --------
    param: [expr] - cross entropy loss
    type: theano scalar

    '''
    # just call theano's version of it, seems reasonable
    return T.mean(T.nnet.categorical_crossentropy(y_pred, y))


def squared_error(y, y_pred):
    ''' basic squared error loss function

    Parameters:
    -----------
    param: y - true labels
    type: theano matrix

    param: y_pred - predicted labels
    type: theano matrix

    Returns:
    --------
    param: [expr] - squared loss
    type: theano scalar
    '''

    return 0.5 * T.mean(T.sum((y - y_pred) ** 2, axis=1))


def sparsity(act, beta=None, rho=None):
    ''' Term used to enforce sparsity in the activations of hidden units for
            autoencoders 

    Parameters:
    -----------
    param: act - activation values
    type: theano matrix

    param: beta - sparsity coefficient
    type: float

    param: rho - sparsity level
    type: float

    '''

    sparse_loss = 0

    if beta is not None and rho is not None:
        avg_act = T.mean(act, axis=0)
        sparse_loss = beta * \
            T.sum(rho * T.log(rho / avg_act) + (1 - rho)
                  * T.log((1 - rho) / (1 - avg_act)))

    return sparse_loss

# debugging
def sparsity_np(act, beta=None, rho=None):

    sparse_loss = 0

    if beta is not None and rho is not None:
        avg_act = np.mean(act, axis=0)
        sparse_loss = beta * \
            np.sum(rho * np.log(rho / avg_act) + (1 - rho)
                  * np.log((1 - rho) / (1 - avg_act)))

    return sparse_loss


