import numpy as np
import theano
import theano.tensor as T
from deepnet.common import nnetact as na
from deepnet.common import nnetloss as nl
from deepnet.common import nnetutils as nu

X = T.matrix('X')
y = T.matrix('y')

# define the architecture
d = 784
k = 10
h1 = 625; activ1 = na.reLU
h2 = 625; activ2 = na.reLU
activ3 = na.softmax

# initialize weights and biases
wts = [None,None,None]
bs = [None,None,None]
np.random.seed(1234)
wts[0] = theano.shared(nu.floatX(0.01*np.random.randn(d,h1)),borrow=True)
wts[1] = theano.shared(nu.floatX(0.01*np.random.randn(h1,h2)),borrow=True)
wts[2] = theano.shared(nu.floatX(0.01*np.random.randn(h2,k)),borrow=True)
bs[0] = theano.shared(nu.floatX(np.zeros(h1)),borrow=True)
bs[1] = theano.shared(nu.floatX(np.zeros(h2)),borrow=True)
bs[2] = theano.shared(nu.floatX(np.zeros(k)),borrow=True)

# forward propagation
act1 = activ1(T.dot(X,wts[0]) + bs[0])
compute_act1 = theano.function(inputs=[X],outputs=act1,allow_input_downcast=True,mode='FAST_RUN')
act2 = activ2(T.dot(act1,wts[1]) + bs[1])
compute_act2 = theano.function(inputs=[X],outputs=act2,allow_input_downcast=True,mode='FAST_RUN')
y_pred = activ3(T.dot(act2,wts[2]) + bs[2])
compute_y_pred = theano.function(inputs=[X],outputs=y_pred,allow_input_downcast=True,mode='FAST_RUN')

X_tr = np.random.randn(128,784)
c_act1 = compute_act1(X_tr)
print c_act1[:5,:5]
