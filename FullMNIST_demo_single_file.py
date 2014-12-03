# Logistic regression
import idx2numpy
import numpy as np
import nnetutils as nu
from theano import function, shared
import theano.tensor as T
rng = np.random

print 'Loading data...'

train_img_path = '/home/bhargav/datasets/MNIST/train-images.idx3-ubyte'
train_lbl_path = '/home/bhargav/datasets/MNIST/train-labels.idx1-ubyte' 
test_img_path = '/home/bhargav/datasets/MNIST/t10k-images.idx3-ubyte' 
test_lbl_path = '/home/bhargav/datasets/MNIST/t10k-labels.idx1-ubyte'

# define training and validation data
train_img = idx2numpy.convert_from_file(train_img_path)
m,row,col = train_img.shape
d = row*col
X = np.reshape(train_img,(m,d)).T/255.

train_lbl = idx2numpy.convert_from_file(train_lbl_path)
k = max(train_lbl)+1
y = np.zeros((k,m))
for i,idx in enumerate(train_lbl):
	y[idx,i] = 1

split = 0.5 # proporition to split for training/validation
pidx = np.random.permutation(m)

m_tr = int(split*m)
X_tr = nu.floatX(X[:,pidx[:m_tr]])
y_tr = nu.floatX(y[:,pidx[:m_tr]])

X_val = nu.floatX(X[:,pidx[m_tr:]])
y_val = nu.floatX(y[:,pidx[m_tr:]])

# set the data matrix for test
test_img = idx2numpy.convert_from_file(test_img_path)
m_te = test_img.shape[0]
X_te = nu.floatX(np.reshape(test_img,(m_te,d)).T/255.) # test data matrix
test_lbl = idx2numpy.convert_from_file(test_lbl_path)

# set the targets for the test-set
y_te = np.zeros((k,m_te))
for i,idx in enumerate(test_lbl):
	y_te[idx,i] = 1
y_te = nu.floatX(y_te)

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.matrix("y")
W = [shared(nu.floatX(rng.randn(k,d)),name="w")] # weight vector
b = [shared(nu.floatX(np.zeros((k,1))),name="b")] # bias vector

# Construct Theano expression graph - this essentially defines the cost function 
# and gradient
print "Building Theano expression graph..."
cost = T.mean(T.sum(-1.0*y*T.log(nu.softmax(T.dot(W[0],x)+b)),axis=0))
gW,gb = T.grad(cost,[W[0],b[0]])
pred = T.argmax(nu.softmax(T.dot(W[0],x)+b),axis=0)
pred_func = T.sum(T.neq(pred,y))

learn_rate = 0.35
# Compile - the main function is one iterative step in training.
# The 'updates' parameter in this context is our optimization routine
print "Compiling.."
train = function(inputs=[x,y],outputs=cost,updates=((W[0],W[0]-learn_rate*gW),(b[0],b[0]-learn_rate*gb)))
predict = function([x,y],pred_func)

# Train
n_iter = 10
for i in range(10):
	cost = train(X_tr,y_tr)

print "Final model:"
print W.get_value(), b.get_value()
