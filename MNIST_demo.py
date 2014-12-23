import idx2numpy
import numpy as np
import nnetutils as nu
import theano
import theano.tensor as T
import MultilayerNet as mln

print 'Loading data...'

train_img_path = '/home/avasbr/datasets/MNIST/train-images.idx3-ubyte'
train_lbl_path = '/home/avasbr/datasets/MNIST/train-labels.idx1-ubyte' 
test_img_path = '/home/avasbr/datasets/MNIST/t10k-images.idx3-ubyte'
test_lbl_path = '/home/avasbr/datasets/MNIST/t10k-labels.idx1-ubyte'

# define training and validation data
train_img = idx2numpy.convert_from_file(train_img_path)
m,row,col = train_img.shape
d = row*col
X = np.reshape(train_img,(m,d))/255.

train_lbl = idx2numpy.convert_from_file(train_lbl_path)
k = max(train_lbl)+1

y = np.zeros((m,k)) # 'one-hot' representation
for i,idx in enumerate(train_lbl):
	y[i,idx] = 1

# set the data matrix for test
test_img = idx2numpy.convert_from_file(test_img_path)
m_te = test_img.shape[0]
X_te = nu.floatX(np.reshape(test_img,(m_te,d))/255.) # test data matrix
test_lbl = nu.floatX(idx2numpy.convert_from_file(test_lbl_path)) 

y_te = np.zeros((m_te,k)) # 'one-hot' representation
for i,idx in enumerate(test_lbl):
	y_te[i,idx] = 1

# Train a model with the same learning rate on the training set, test on the testing set:
print 'Training...'

mln_params = {'d':d,'k':k,'num_hid':[1024,1024],'activ':[nu.reLU,nu.reLU,nu.softmax],'loss_func':nu.cross_entropy,
'dropout_flag':True,'input_p':0.2,'hidden_p':0.5}

# various methods to try - simply change what goes into the fit function
sgd_params = {'method':'SGD','num_epochs':100,'batch_size':128,'learn_rate':0.5}
rmsprop_params = {'method':'RMSPROP','num_epochs':100,'batch_size':128,'learn_rate':0.001,
'rho':0.9,'max_norm':True,'c':15}
adagrad_params = {'method':'ADAGRAD','num_epochs':100,'batch_size':128,'learn_rate':1.,'max_norm':False,'c':15.0}

nnet = mln.MultilayerNet(**mln_params)
nnet.fit(X,y,**rmsprop_params)

print 'Performance on test set:'
print 100*nnet.score(X_te,y_te),'%'