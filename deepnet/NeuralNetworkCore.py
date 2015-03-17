import cPickle
import os
import sys
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from deepnet.common import nnetutils as nu
from deepnet.common import nnetloss as nl
from deepnet.common import nnetact as na
from deepnet.common import nnetoptim as nopt
from deepnet.common import nneterror as ne


class Network(object):

    ''' Core neural network class that forms the basis for all further implementations (e.g.
        MultilayerNet, Autoencoder, etc). Contains basic functions for propagating data forward
        and backwards through the network, as well as fitting the weights to data'''

    def __init__(self, d=None, k=None, num_hids=None, activs=None, loss_terms=[None], **loss_params):

        # Number of units in the output layer determined by k, so not explicitly specified in
        # num_hids. still need to check that there's one less hidden layer than number of activation
        # functions
        assert(len(num_hids) + 1 == len(activs))

        # number of nodes
        self.num_nodes = [d] + num_hids + [k]

        # total number of parameters in this neural network
        self.num_params = 0
        for i, (n1, n2) in enumerate(zip(self.num_nodes[:-1], self.num_nodes[1:])):
            self.num_params += (n1 + 1) * n2

        # define activation functions
        self.activs = [None] * len(activs)
        for idx, activ in enumerate(activs):
            if activ == 'sigmoid':
                self.activs[idx] = na.sigmoid
            elif activ == 'tanh':
                self.activs[idx] = na.tanh
            elif activ == 'reLU':
                self.activs[idx] = na.reLU
            elif activ == 'softmax':
                self.activs[idx] = na.softmax
            else:
                sys.exit(ne.activ_err())

        # loss function and parameters
        self.loss_terms = loss_terms
        self.loss_params = loss_params

        # initialize the random number stream
        self.srng = RandomStreams()
        self.srng.seed(np.random.randint(99999))

    def set_weights(self, wts=None, bs=None, init_method=None, scale_factor=None, seed=None):
        ''' Initializes the weights and biases of the neural network

        Parameters:
        -----------
        param: wts - weights
        type: np.ndarray, optional

        param: bs - biases
        type: np.ndarray, optional

        param: init_method - calls some pre-specified weight initialization routines
        type: string

        param: scale_factor - for gauss, corresponds to the standard deviation
        type: float, optional
        '''
        if seed is not None:
            np.random.seed(seed=seed)
            self.srng.seed(seed)

        # weights and biases
        if wts is None and bs is None:
            wts = (len(self.num_nodes) - 1) * [None]
            bs = (len(self.num_nodes) - 1) * [None]

            if init_method == 'gauss':
                for i, (n1, n2) in enumerate(zip(self.num_nodes[:-1], self.num_nodes[1:])):
                    wts[i] = scale_factor * 1. / \
                        np.sqrt(n2) * np.random.randn(n1, n2)
                    bs[i] = np.zeros(n2)

            elif init_method == 'fan-io':
                for i, (n1, n2) in enumerate(zip(self.num_nodes[:-1], self.num_nodes[1:])):
                    v = scale_factor * np.sqrt(6. / (n1 + n2 + 1))
                    wts[i] = 2.0 * v * np.random.rand(n1, n2) - v
                    bs[i] = np.zeros(n2)
            else:
                sys.exit(ne.weight_error())

        else:
            assert isinstance(wts, list)
            assert isinstance(bs, list)

        self.wts_ = [theano.shared(nu.floatX(wt), borrow=True) for wt in wts]
        self.bs_ = [theano.shared(nu.floatX(b), borrow=True) for b in bs]

    def fit(self, X_tr, y_tr, X_val=None, y_val=None, wts=None, bs=None, plotting=False, **optim_params):
        ''' The primary function which ingests data and fits to the neural network.
        Currently only supports mini-batch training.

        Parameters:
        -----------
        param: X_tr - training data
        type: theano matrix

        param: y_tr - training labels
        type: theano matrix

        param: X_val - validation data
        type: theano matrix

        param: y_val - validation labels
        type: theano matrix

        param: **optim_params
        type: dictionary of optimization parameters

        '''
        # initialize all the weights
        if all(node for node in self.num_nodes):
            init_method = optim_params.pop('init_method')
            scale_factor = optim_params.pop('scale_factor')
            try:
                seed = optim_params.pop('seed')
            except KeyError:
                seed = None
            self.set_weights(
                wts=wts, bs=bs, init_method=init_method, scale_factor=scale_factor, seed=seed)

        try:
            optim_type = optim_params.pop('optim_type')
        except KeyError:
            sys.exit(ne.opt_type_err())

        # perform minibatch or full-batch optimization
        num_epochs = optim_params.pop('num_epochs', None)
        batch_size = optim_params.pop('batch_size', None)

        if optim_type == 'minibatch':
            self.minibatch_optimize(X_tr, y_tr, X_val=X_val, y_val=y_val, batch_size=batch_size, num_epochs=num_epochs,
                                    plotting=plotting, **optim_params)
        elif optim_type == 'fullbatch':
            self.fullbatch_optimize(
                X_tr, y_tr, X_val=X_val, y_val=y_val, num_epochs=num_epochs, **optim_params)
        else:
            # error
            sys.exit(ne.opt_type_err())

        return self

    def shared_dataset(self, X, y):
        ''' As per the deep learning tutorial, loading the data all at once (if possible)
        into the GPU will significantly speed things up '''

        return theano.shared(nu.floatX(X)), theano.shared(nu.floatX(y))

    def fullbatch_optimize(self, X_tr, y_tr, X_val=None, y_val=None, num_epochs=None, **optim_params):
        ''' Full-batch optimization using scipy's L-BFGS-B and CG

        Parameters:
        -----------
        param: X_tr - training data
        type: theano matrix

        param: y_tr - training labels
        type: theano matrix

        param: num_epochs - the number of full runs through the dataset
        type: int
        '''

        X = T.matrix('X')  # input variable
        y = T.matrix('y')  # output variable
        w = T.vector('w')  # weight vector

        # reshape w into wts/biases
        wts, bs = nu.t_reroll(w, self.num_nodes)

        # get the loss
        optim_loss = self.compute_optim_loss(X, y, wts=wts, bs=bs)

        # compute grad
        params = [p for param in [wts, bs]
                  for p in param]  # all model parameters in a list
        # gradient of each model param w.r.t training loss
        grad_params = [T.grad(optim_loss, param) for param in params]
        # gradient of the full weight vector
        grad_w = nu.t_unroll(grad_params[:len(wts)], grad_params[len(wts):])

        compute_loss_grad_from_vector = theano.function(
            inputs=[w, X, y],
            outputs=[optim_loss, grad_w],
            allow_input_downcast=True)

        compute_loss_from_vector = theano.function(
            inputs=[w, X, y],
            outputs=[optim_loss],
            allow_input_downcast=True)

        # initial value for the weight vector
        wts0 = [wt.get_value() for wt in self.wts_]
        bs0 = [b.get_value() for b in self.bs_]
        w0 = nu.unroll(wts0, bs0)

        # print 'Checking gradients for fun...'
        # self.check_gradients(X_tr,y_tr,wts0,bs0)
        # print 'Pre-training loss:',compute_loss_from_vector(w0,X_tr,y_tr)

        try:
            optim_method = optim_params.pop('optim_method')
        except KeyError:
            sys.exit(ne.method_err())

        # very annoying.
        if optim_method == 'L-BFGS-B' and theano.config.floatX == 'float32':
            sys.exit('Sorry, L-BFGS-B only works with float64')

        # scipy optimizer
        wf = sp.optimize.minimize(compute_loss_grad_from_vector, w0, args=(X_tr, y_tr), method=optim_method, jac=True,
                                  options={'maxiter': num_epochs})

        # print 'Post-training loss',compute_loss_from_vector(wf.x,X_tr,y_tr)

        # re-roll this back into weights and biases
        wts, bs = nu.reroll(wf.x, self.num_nodes)

        self.wts_ = [theano.shared(nu.floatX(wt)) for wt in wts]
        self.bs_ = [theano.shared(nu.floatX(b)) for b in bs]

    def minibatch_optimize(self, X_tr, y_tr, X_val=None, y_val=None, batch_size=None, num_epochs=None, plotting=False, **optim_params):
        ''' Mini-batch optimization using update functions; however, if the batch size = m, then this is basically
        full-batch learning with gradient descent

        Parameters:
        -----------
        param: X_tr - training data
        type: theano matrix

        param: y_tr - training labels
        type: theano matrix

        param: updates - update per rule for each

        param: batch_size - number of examples per mini-batch
        type: int

        param: num_epochs - the number of full runs through the dataset
        type: int

        '''
        X = T.matrix('X')  # input variable
        y = T.matrix('y')  # output variable
        idx = T.ivector('idx')  # integer index

        optim_loss = self.compute_optim_loss(X, y)  # optimization loss
        eval_loss = self.compute_eval_loss(X, y)  # evaluation loss
        params = [p for param in [self.wts_, self.bs_]
                  for p in param]  # all model parameters in a list
        # gradient of each model param w.r.t training loss
        grad_params = [T.grad(optim_loss, param) for param in params]

        # get the method and learning type
        try:
            optim_method = optim_params.pop('optim_method')
        except KeyError:
            sys.exit(ne.method_err())

        # define the update rule
        updates = []
        if optim_method == 'SGD':
            updates = nopt.sgd(
                params, grad_params, **optim_params)  # update rule

        elif optim_method == 'ADAGRAD':
            updates = nopt.adagrad(
                params, grad_params, **optim_params)  # update rule

        elif optim_method == 'RMSPROP':
            updates = nopt.rmsprop(params, grad_params, **optim_params)

        else:
            print method_err()

        # define the mini-batches
        m = X_tr.shape[0]  # total number of training instances
        # number of batches, based on batch size
        n_batches = int(m / batch_size)
        # batch_size won't divide the data evenly, so get leftover
        leftover = m - n_batches * batch_size

        # load the full dataset into a shared variable - this is especially useful
        # for test
        X_tr, y_tr = self.shared_dataset(X_tr, y_tr)

        # training function for minibatchs
        train = theano.function(
            inputs=[idx],
            updates=updates,
            allow_input_downcast=True,
            mode='FAST_RUN',
            givens={
                X: X_tr[idx],
                y: y_tr[idx]
            })

        compute_train_loss = theano.function(
            inputs=[],
            outputs=eval_loss,
            allow_input_downcast=True,
            mode='FAST_RUN',
            givens={
                X: X_tr,
                y: y_tr
            })

        # if validation data is provided, validation loss
        compute_val_loss = None

        if X_val is not None and y_val is not None:
            X_val, y_val = self.shared_dataset(X_val, y_val)
            compute_val_loss = theano.function(
                inputs=[],
                outputs=eval_loss,
                allow_input_downcast=True,
                mode='FAST_RUN',
                givens={
                    X: X_val,
                    y: y_val
                })

        # iterate through the training examples
        tr_loss = []
        val_loss = []
        epoch = 0

        while epoch < num_epochs:
            # randomly shuffle the data indices
            tr_idx = np.random.permutation(m)
            # define the start-stop indices
            ss_idx = range(0, m + 1, batch_size)
            ss_idx[-1] += leftover  # add the leftovers to the last batch

            # run through a full epoch
            for idx, (start_idx, stop_idx) in enumerate(zip(ss_idx[:-1], ss_idx[1:])):

                # total number of batches processed up until now
                n_batch_iter = (epoch - 1) * n_batches + idx
                batch_idx = tr_idx[start_idx:stop_idx]  # get the next batch

                train(batch_idx)

            epoch += 1  # update the epoch count
            if epoch % 10 == 0:
                tr_loss.append(compute_train_loss())
                if compute_val_loss is not None:
                    val_loss.append(compute_val_loss())
                    print 'Epoch: %s, Training error: %.15f, Validation error: %.15f' % (epoch, tr_loss[-1], val_loss[-1])
                else:
                    print 'Epoch: %s, Training error: %.15f' % (epoch, tr_loss[-1])

            # training and validation curves - very useful to see how training
            # error evolves
            if plotting:
                num_pts = len(tr_loss)
                pts = [idx * 10 for idx in range(num_pts)]
                plt.plot(pts, tr_loss, label='Training loss')
                # sort of a weak way to check if validation losses have been
                # computed
                if len(val_loss) > 0:
                    plt.plot(pts, val_loss, label='Validation loss')

                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend(loc='upper right')
                plt.show()

    def dropout(self, act, p=0.5):
        ''' Randomly drops an activation with probability p 

        Parameters
        ----------
        param: act - activation values, in a matrix
        type: theano matrix

        param: p - probability of dropping out a node
        type: float, optional

        Returns:
        --------
        param: [expr] - activation values randomly zeroed out
        type: theano matrix

        '''
        if p > 0:
            # randomly dropout p activations
            retain_prob = 1. - p
            return (1. / retain_prob) * act * self.srng.binomial(act.shape, p=retain_prob, dtype=theano.config.floatX)

    def train_fprop(self, X, wts=None, bs=None):
        ''' Performs forward propagation with for training, which could be different from
        the vanilla frprop we would use for testing, due to extra bells and whistles such as 
        dropout, corruption, etc'''

        if wts is None and bs is None:
            wts = self.wts_
            bs = self.bs_

        if 'dropout' in self.loss_terms:
            input_p = self.loss_params['input_p']
            hidden_p = self.loss_params['hidden_p']

            # compute the first activation separately in case we have no hidden
            # layer;
            act = self.activs[0](
                T.dot(self.dropout(X, input_p), wts[0]) + bs[0])
            if len(wts) > 1:  # len(wts) = 1 corresponds to softmax regression
                for i, (w, b, activ) in enumerate(zip(wts[1:], bs[1:], self.activs[1:])):
                    act = activ(T.dot(self.dropout(act, hidden_p), w) + b)

            eps = 1e-6
            act = T.switch(act < eps, eps, act)
            act = T.switch(act > (1. - eps), (1. - eps), act)

            return act
        else:
            return self.fprop(X, wts, bs)

    def fprop(self, X, wts=None, bs=None):
        ''' Performs vanilla forward propagation through the network

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
        param: act - final activation values
        type: theano matrix
        '''
        if wts is None and bs is None:
            wts = self.wts_
            bs = self.bs_

        # use the first data matrix to compute the first activation
        act = self.activs[0](T.dot(X, wts[0]) + bs[0])
        
        # len(wts) = 1 corresponds to softmax regression
        if len(wts) > 1:
            for i, (w, b, activ) in enumerate(zip(wts[1:], bs[1:], self.activs[1:])):
                act = activ(T.dot(act, w) + b)

        # for numericaly stability
        eps = 1e-6
        act = T.switch(act < eps, eps, act)
        act = T.switch(act > (1. - eps), (1. - eps), act)

        return act

    def check_gradients(self, X_in, Y_in, wts=None, bs=None):
        ''' this seems like overkill, but I suppose it doesn't hurt to have it in here...'''

        # assume that if it's not provided, they will be shared variables - this is
        # probably dangerous, but this is a debugging tool anyway,
        # so...whatever
        if wts is None and bs is None:
            wts = self.wts_
            bs = self.bs_
        else:
            wts = [theano.shared(nu.floatX(w), borrow=True) for w in wts]
            bs = [theano.shared(nu.floatX(b), borrow=True) for b in bs]

        X = T.matrix()  # inputs
        Y = T.matrix()  # labels
        v = T.vector()  # vector of biases and weights
        i = T.lscalar()  # index

        # 1. compile the numerical gradient function
        def compute_numerical_gradient(v, i, X, Y, eps=1e-4):

            # perturb the input
            v_plus = T.inc_subtensor(v[i], eps)
            v_minus = T.inc_subtensor(v[i], -1.0 * eps)

            # roll it back into the weight matrices and bias vectors
            wts_plus, bs_plus = nu.t_reroll(v_plus, self.num_nodes)
            wts_minus, bs_minus = nu.t_reroll(v_minus, self.num_nodes)

            # compute the loss for both sides, and then compute the numerical
            # gradient
            loss_plus = self.compute_optim_loss(X, Y, wts=wts_plus, bs=bs_plus)
            loss_minus = self.compute_optim_loss(X, Y, wts_minus, bs_minus)

            # ( E(weights[i]+eps) - E(weights[i]-eps) )/(2*eps)
            return 1.0 * (loss_plus - loss_minus) / (2 * eps)

        compute_ngrad = theano.function(
            inputs=[v, i, X, Y], outputs=compute_numerical_gradient(v, i, X, Y))

        # 2. compile backprop (theano's autodiff)
        optim_loss = self.compute_optim_loss(X, Y, wts=wts, bs=bs)
        params = [p for param in [wts, bs]
                  for p in param]  # all model parameters in a list
        # gradient of each model param w.r.t training loss
        grad_params = [T.grad(optim_loss, param) for param in params]
        # gradient of the full weight vector
        grad_w = nu.t_unroll(grad_params[:len(wts)], grad_params[len(wts):])

        compute_bgrad = theano.function(inputs=[X, Y], outputs=grad_w)

        # compute the mean difference between the numerical and exact gradients
        v0 = nu.unroll([wt.get_value()
                        for wt in wts], [b.get_value() for b in bs])
        # get the indices of the weights/biases we want to check
        idxs = np.random.permutation(self.num_params)[:(self.num_params / 5)]

        ngrad = [None] * len(idxs)
        for j, idx in enumerate(idxs):
            ngrad[j] = compute_ngrad(v0, idx, X_in, Y_in)
        bgrad = compute_bgrad(X_in, Y_in)[idxs]

        cerr = np.mean(np.abs(ngrad - bgrad))
        assert cerr < 1e-10

    def compute_eval_loss(self, X, y, wts=None, bs=None):
        ''' Given inputs, returns the evaluation loss at the current state of the model

        Parameters:
        -----------
        param: X - training data
        type: theano matrix

        param: y - training labels
        type: theano matrix

        param: wts - weights
        type: theano matrix, optional

        param: bs - biases
        type: theano matrix, optional

        Returns:
        --------
        param: eval_loss - evaluation loss, which doesn't include regularization
        type: theano scalar

        '''
        if wts is None and bs is None:
            wts = self.wts_
            bs = self.bs_

        eval_loss = None  # the loss function we can evaluate during validation
        y_pred = self.fprop(X, wts, bs)

        if 'cross_entropy' in self.loss_terms:
            eval_loss = nl.cross_entropy(y, y_pred)

        elif 'binary_cross_entropy' in self.loss_terms:
            eval_loss = nl.binary_cross_entropy(y, y_pred)

        elif 'squared_error' in self.loss_terms:
            eval_loss = nl.squared_error(y, y_pred)
        else:
            sys.exit('Must be either cross_entropy or squared_error')

        return eval_loss

    def compute_optim_loss(self, X, y, wts=None, bs=None):
        ''' Given inputs, returns the training loss at the current state of the model

        Parameters:
        -----------
        param: X - training data
        type: theano matrix

        param: y - training labels
        type: theano matrix

        param: wts - weights
        type: theano matrix, optional

        param: bs - biases
        type: theano matrix, optional

        Returns:
        --------
        param: optim_loss - the optimization loss which must be optimized over
        type: theano scalar
        '''
        if wts is None and bs is None:
            wts = self.wts_
            bs = self.bs_

        y_optim = self.train_fprop(X, wts, bs)
        # the loss function which will specifically be optimized over
        optim_loss = None

        if 'cross_entropy' in self.loss_terms:
            optim_loss = nl.cross_entropy(y, y_optim)

        elif 'binary_cross_entropy' in self.loss_terms:
            optim_loss = nl.binary_cross_entropy(y, y_optim)

        elif 'squared_error' in self.loss_terms:
            optim_loss = nl.squared_error(y, y_optim)

        else:
            sys.exit('Must be either cross_entropy or squared_error')

        if 'l1_reg' in self.loss_terms:
            l1_decay = self.loss_params.get('l1_decay')
            optim_loss += nl.l1_reg(wts, l1_decay=l1_decay)

        if 'l2_reg' in self.loss_terms:
            l2_decay = self.loss_params.get('l2_decay')
            optim_loss += nl.l2_reg(wts, l2_decay=l2_decay)

        return optim_loss

    def get_weights_and_biases(self):
        ''' simple function which returns the weights and biases as numpy arrays'''

        wts = [wt.get_value() for wt in self.wts_]
        bs = [b.get_value() for b in self.bs_]

        return wts, bs

    # debugging
    def check_nans(self):
        ''' simple function which returns True if any value is NaN in wts or biases '''

        # poke into the shared variables and get their values
        wts, bs = self.get_weights_and_biases()
        nans = 0
        for wt, b in zip(wts, bs):
            nans += np.sum(wt) + np.sum(b)

        return np.isnan(nans)
