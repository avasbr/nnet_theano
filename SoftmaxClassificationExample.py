# Logistic regression
import numpy
from theano import function, shared
import theano.tensor as T
import numpy as np
import idx2numpy

# Define paths to the data
print "Loading data..."
train_img_path = '/home/bhargav/datasets/MNIST/train-images.idx3-ubyte'
train_lbl_path = '/home/bhargav/datasets/MNIST/train-labels.idx1-ubyte' 
test_img_path = '/home/bhargav/datasets/MNIST/t10k-images.idx3-ubyte' 
test_lbl_path = '/home/bhargav/datasets/MNIST/t10k-labels.idx1-ubyte'

# convert the raw images into feature vectors
train_img = idx2numpy.convert_from_file(train_img_path)
m_tr,row,col = train_img.shape
d = row*col # dimensions
X_tr = np.reshape(train_img,(m_tr,d)).T/255. # train data matrix
y_lbl = idx2numpy.convert_from_file(train_lbl_path)
k = max(y_lbl)+1

# set the targets for the training-set
y_tr = np.zeros((k,m_tr))
for i,idx in enumerate(y_lbl):
	y_tr[idx,i] = 1

# set the data matrix for test
test_img = idx2numpy.convert_from_file(test_img_path)
m_te = test_img.shape[0]
X_te = np.reshape(test_img,(m_te,d)).T/255. # test data matrix
y_lbl = idx2numpy.convert_from_file(test_lbl_path)

# Declare Theano symbolic variables
X = T.dmatrix("X")
Y = T.dmatrix("Y")

# initialize the 
W = shared(np.random.randn(d,k),name="W") # weight matrix
b = shared(np.zeros((k,1),dtype='float64'),name="b",broadcastable=(False,True)) # bias vector

# construct Theano expression graph - this essentially defines the cost function 
# and gradient
decay = 0.01
learn_rate = 0.1
n_iter = 1000

# softmax function - T.nnet.softmax is also an option
Z = T.dot(W.T,X) + b # this should naturally broadcast
max_v = T.max(Z,axis=0,keepdims=True)
log_sum = T.log(T.sum(T.exp(Z-max_v),axis=0)) + max_v
prob = T.exp(Z-log_sum)

# cost function
cost = T.mean(T.sum(-1.0*Y*T.log(prob),axis=0)) + 0.5*decay*(W**2).sum()
gW,gb = T.grad(cost,[W,b]) # reverse-mode

# compile functions
# the train function takes in an instance of X,y,and updates the weight and bias parameters - could
# probably add in validation sets for this too
print "Compiling.."
train = function([X,Y],updates=[(W,W-learn_rate*gW),(b,b-learn_rate*gb)])
predict = function(inputs=[X],outputs=T.argmax(prob))

print "Training..."
for i in range(n_iter):
	train(X_tr,y_tr)

print "Evaluating..."
pred = predict(X_te)
acc = 100.*np.mean(pred==y_lbl)
print acc