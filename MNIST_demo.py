import idx2numpy
import numpy as np
import nnetutils as nu
import theano
import theano.tensor as T
import MultilayerNet as mln

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

print 'Training...'

mln_params = {'d':d,'k':k,'n_hid':[],'activ':[nu.softmax],'cost_type':'cross_entropy','L2_decay':0.0}
optim_params = {'method':'SGD','n_iter':100,'lr':0.35}
nnet = mln.MultilayerNet(**mln_params)
nnet.fit(X_tr,y_tr,**optim_params)
pred,mce = nnet.predict(X_te,y_te)

print 'Performance on test set:'
print 'Accuracy:',100.*(1-mce),'%'